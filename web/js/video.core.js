/**
 * VHS (Video Helper Suite) 精简版核心模块
 * 只包含视频预览和文件上传功能
 */

import { app } from '../../../scripts/app.js'
import { api } from '../../../scripts/api.js'

/**
 * 链式回调函数 - 将新的回调函数链接到现有属性上
 * @param {Object} object - 目标对象
 * @param {string} property - 属性名
 * @param {Function} callback - 新的回调函数
 */
function chainCallback(object, property, callback) {
    if (object == undefined) {
        // 这不应该发生
        console.error("Tried to add callback to non-existant object")
        return;
    }
    if (property in object && object[property]) {
        // 如果属性已存在，将新回调链接到原有回调之后
        const callback_orig = object[property]
        object[property] = function () {
            const r = callback_orig.apply(this, arguments);
            return callback.apply(this, arguments) ?? r
        };
    } else {
        // 如果属性不存在，直接设置回调
        object[property] = callback;
    }
}

/**
 * 自动调整节点高度以适应内容
 * @param {Object} node - 节点对象
 */
function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]])
    node?.graph?.setDirtyCanvas(true);
}

/**
 * 开始拖拽操作
 * @param {Object} node - 节点对象
 * @param {Object} pointer - 指针对象
 */
function startDraggingItems(node, pointer) {
    app.canvas.emitBeforeChange()
    app.canvas.graph?.beforeChange()
    // 确保拖拽在成功或失败时都能正确清理
    pointer.finally = () => {
      app.canvas.isDragging = false
      app.canvas.graph?.afterChange()
      app.canvas.emitAfterChange()
    }
    app.canvas.processSelect(node, pointer.eDown, true)
    app.canvas.isDragging = true
}

/**
 * 处理拖拽结束后的操作
 * @param {Object} e - 事件对象
 */
function processDraggedItems(e) {
    if (e.shiftKey || LiteGraph.alwaysSnapToGrid)
      app.graph?.snapToGrid(app.canvas.selectedItems)
    app.canvas.dirty_canvas = true
    app.canvas.dirty_bgcanvas = true
    app.canvas.onNodeMoved?.(findFirstNode(app.canvas.selectedItems))
}

/**
 * 允许从组件开始拖拽节点
 * @param {Object} widget - 组件对象
 */
function allowDragFromWidget(widget) {
    widget.onPointerDown = function(pointer, node) {
        pointer.onDragStart = () => startDraggingItems(node, pointer)
        pointer.onDragEnd = processDraggedItems
        app.canvas.dirty_canvas = true
        return true
    }
}

/**
 * 上传文件到服务器
 * @param {File} file - 要上传的文件
 * @returns {Promise} 上传结果
 */
async function uploadFile(file) {
    try {
        // 将文件包装在 FormData 中，以便包含文件名
        const body = new FormData();
        const i = file.webkitRelativePath ? file.webkitRelativePath.lastIndexOf('/') : -1;
        const subfolder = i > 0 ? file.webkitRelativePath.slice(0,i+1) : "";
        const new_file = new File([file], file.name, {
            type: file.type,
            lastModified: file.lastModified,
        });
        body.append("image", new_file);
        // 默认上传到 input 目录，确保 /view 可访问
        body.append("type", "input");
        if (i > 0) {
            body.append("subfolder", subfolder);
        }
        
        console.log(`上传文件: ${file.name}, 子文件夹: ${subfolder}`);
        const resp = await api.fetchApi("/upload/image", {
            method: "POST",
            body,
        });

        if (resp.status === 200) {
            console.log(`文件上传成功: ${file.name}`);
            return resp
        } else {
            console.error(`文件上传失败: ${file.name}, 状态: ${resp.status}, 原因: ${resp.statusText}`);
            return { status: resp.status, statusText: resp.statusText };
        }
    } catch (error) {
        console.error(`文件上传异常: ${file.name}`, error);
        return { status: 500, statusText: error.message };
    }
}

/**
 * 为节点添加文件上传功能
 * @param {Object} nodeType - 节点类型
 * @param {Object} nodeData - 节点数据
 * @param {string} widgetName - 要关联的组件名称
 * @param {string} type - 上传类型（"video", "audio", "folder"）
 */
function addUploadWidget(nodeType, nodeData, widgetName, type="video") {
    chainCallback(nodeType.prototype, "onNodeCreated", function() {
        const pathWidget = this.widgets.find((w) => w.name === widgetName);
        const fileInput = document.createElement("input");
        chainCallback(this, "onRemoved", () => {
            fileInput?.remove();
        });
        
        if (type == "folder") {
            Object.assign(fileInput, {
                type: "file",
                style: "display: none",
                webkitdirectory: true,
                onchange: async () => {
                    try {
                    const directory = fileInput.files[0].webkitRelativePath;
                    const i = directory.lastIndexOf('/');
                    if (i <= 0) {
                            alert("无法找到文件夹路径，请确保选择了文件夹而不是单个文件");
                            return;
                    }
                    const path = directory.slice(0,directory.lastIndexOf('/'))
                        // 确保 options.values 存在，如果不存在则初始化为空数组
                        if (!pathWidget.options.values) {
                            pathWidget.options.values = [];
                        }
                    if (pathWidget.options.values.includes(path)) {
                            alert("同名文件夹已存在");
                        return;
                    }
                    let successes = 0;
                        let totalFiles = fileInput.files.length;
                        console.log(`开始上传文件夹: ${path}, 包含 ${totalFiles} 个文件`);
                        
                    for(const file of fileInput.files) {
                            console.log(`上传文件: ${file.webkitRelativePath}`);
                            const uploadResult = await uploadFile(file);
                            if (uploadResult && uploadResult.status == 200) {
                            successes++;
                                console.log(`文件上传成功: ${file.name}`);
                        } else {
                                console.error(`文件上传失败: ${file.name}, 状态: ${uploadResult?.status}`);
                                // 上传失败，但之前的上传可能已经成功
                                // 停止后续上传以防止级联失败
                                // 只有在至少一个上传成功时才添加到列表
                            if (successes > 0) {
                                break
                            } else {
                                    alert(`文件夹上传失败，没有文件成功上传`);
                                return;
                            }
                        }
                    }
                        
                        console.log(`文件夹上传完成: ${path}, 成功上传 ${successes}/${totalFiles} 个文件`);
                    pathWidget.options.values.push(path);
                    pathWidget.value = path;
                    if (pathWidget.callback) {
                        pathWidget.callback(path)
                        }
                        alert(`文件夹上传成功: ${path} (${successes}/${totalFiles} 个文件)`);
                    } catch (error) {
                        console.error("文件夹上传过程中发生错误:", error);
                        alert("文件夹上传过程中发生错误: " + error.message);
                    }
                },
            });
        } else if (type == "video") {
            Object.assign(fileInput, {
                type: "file",
                accept: "video/webm,video/mp4,video/x-matroska,image/gif",
                style: "display: none",
                onchange: async () => {
                    if (fileInput.files.length) {
                        let resp = await uploadFile(fileInput.files[0])
                        if (resp.status != 200) {
                            // 上传失败，文件无法添加到选项
                            return;
                        }
                        const filename = (await resp.json()).name;
                        // 确保 options.values 存在，如果不存在则初始化为空数组
                        if (!pathWidget.options.values) {
                            pathWidget.options.values = [];
                        }
                        pathWidget.options.values.push(filename);
                        pathWidget.value = filename;
                        if (pathWidget.callback) {
                            pathWidget.callback(filename)
                        }
                    }
                },
            });
        } else if (type == "audio") {
            Object.assign(fileInput, {
                type: "file",
                accept: "audio/mpeg,audio/wav,audio/x-wav,audio/ogg",
                style: "display: none",
                onchange: async () => {
                    if (fileInput.files.length) {
                        let resp = await uploadFile(fileInput.files[0])
                        if (resp.status != 200) {
                            // 上传失败，文件无法添加到选项
                            return;
                        }
                        const filename = (await resp.json()).name;
                        // 确保 options.values 存在，如果不存在则初始化为空数组
                        if (!pathWidget.options.values) {
                            pathWidget.options.values = [];
                        }
                        pathWidget.options.values.push(filename);
                        pathWidget.value = filename;
                        if (pathWidget.callback) {
                            pathWidget.callback(filename)
                        }
                    }
                },
            });
        } else {
            throw "Unknown upload type"
        }
        
        document.body.append(fileInput);
        let uploadWidget = this.addWidget("button", "choose " + type + " to upload", "image", () => {
            // 清除当前点击事件
            app.canvas.node_widget = null
            fileInput.click();
        });
        uploadWidget.options.serialize = false;
    });
}

/**
 * 为节点添加视频预览功能
 * @param {Object} nodeType - 节点类型
 * @param {boolean} isInput - 是否为输入节点
 */
function addVideoPreview(nodeType, isInput=true) {
    chainCallback(nodeType.prototype, "onNodeCreated", function() {
        var element = document.createElement("div");
        const previewNode = this;
        var previewWidget = this.addDOMWidget("videopreview", "preview", element, {
            serialize: false,
            hideOnZoom: false,
            getValue() {
                return element.value;
            },
            setValue(v) {
                element.value = v;
            },
        });
        
        allowDragFromWidget(previewWidget)
        
        previewWidget.computeSize = function(width) {
            if (this.aspectRatio && !this.parentEl.hidden) {
                let height = (previewNode.size[0]-20)/ this.aspectRatio + 10;
                if (!(height > 0)) {
                    height = 0;
                }
                this.computedHeight = height + 10;
                return [width, height];
            }
            return [width, -4];
        }
        
        previewWidget.value = {
            hidden: false, 
            paused: false, 
            params: {},
            muted: app.ui.settings.getSettingValue("VideoPreview.DefaultMute")
        }
        
        previewWidget.parentEl = document.createElement("div");
        previewWidget.parentEl.className = "vhs_preview";
        previewWidget.parentEl.style['width'] = "100%"
        previewWidget.parentEl.style['position'] = "relative"
        element.appendChild(previewWidget.parentEl);
        
        // 创建视频元素
        previewWidget.videoEl = document.createElement("video");
        previewWidget.videoEl.controls = false;
        previewWidget.videoEl.loop = true;
        previewWidget.videoEl.muted = true;
        previewWidget.videoEl.style['width'] = "100%"
        // 让覆盖层可以拦截鼠标事件
        previewWidget.videoEl.style.pointerEvents = "none"

        previewWidget.videoEl.addEventListener("loadedmetadata", () => {
            previewWidget.aspectRatio = previewWidget.videoEl.videoWidth / previewWidget.videoEl.videoHeight;
            fitHeight(this);
        });
        
        previewWidget.videoEl.addEventListener("error", () => {
            previewWidget.parentEl.hidden = true;
            fitHeight(this);
        });
        
        previewWidget.videoEl.onmouseenter =  () => {
            previewWidget.videoEl.muted = previewWidget.value.muted
        };
        previewWidget.videoEl.onmouseleave = () => {
            previewWidget.videoEl.muted = true;
        };

        // 创建图像元素
        previewWidget.imgEl = document.createElement("img");
        previewWidget.imgEl.style['width'] = "100%"
        previewWidget.imgEl.hidden = true;
        previewWidget.imgEl.onload = () => {
            previewWidget.aspectRatio = previewWidget.imgEl.naturalWidth / previewWidget.imgEl.naturalHeight;
            fitHeight(this);
        };
        
        var timeout = null;
        this.updateParameters = (params, force_update) => {
            if (!previewWidget.value.params) {
                if(typeof(previewWidget.value) != 'object') {
                    previewWidget.value =  {hidden: false, paused: false}
                }
                previewWidget.value.params = {}
            }
            Object.assign(previewWidget.value.params, params)
            
            if (!force_update &&
                app.ui.settings.getSettingValue("VideoPreview.AdvancedPreviews") == 'Never') {
                return;
            }
            
            if (timeout) {
                clearTimeout(timeout);
            }
            if (force_update) {
                previewWidget.updateSource();
            } else {
                timeout = setTimeout(() => previewWidget.updateSource(),100);
            }
        };
        
        previewWidget.updateSource = function () {
            if (this.value.params == undefined) {
                return;
            }
            
            let params =  {}
            let advp = app.ui.settings.getSettingValue("VideoPreview.AdvancedPreviews")
            if (advp == 'Never') {
                advp = false
            } else if (advp == 'Input Only') {
                advp = isInput
            } else {
                advp = true
            }
            
            Object.assign(params, this.value.params); // 浅拷贝
            params.timestamp = Date.now()
            this.parentEl.hidden = this.value.hidden;
            
            // 如果前端不存在 VHS 的路径小部件，视为 VHS 功能不可用（后端多半没有 /vhs 路由）
            const hasVHS = !!(app.widgets && app.widgets.VHSPATH);
            const useVHS = advp && hasVHS;

            if (params.format?.split('/')[0] == 'video'
                || advp && (params.format?.split('/')[1] == 'gif')
                || params.format == 'folder') {

                this.videoEl.autoplay = !this.value.paused && !this.value.hidden;
                if (!useVHS) {
                    this.videoEl.src = api.apiURL('/view?' + new URLSearchParams(params));
                } else {
                    let target_width = (previewNode.size[0]-20)*2 || 256;
                    let minWidth = app.ui.settings.getSettingValue("VideoPreview.AdvancedPreviewsMinWidth")
                    if (target_width < minWidth) {
                        target_width = minWidth
                    }
                    if (!params.custom_width || !params.custom_height) {
                        params.force_size = target_width+"x?"
                    } else {
                        let ar = params.custom_width/params.custom_height
                        params.force_size = target_width+"x"+(target_width/ar)
                    }
                    params.deadline = app.ui.settings.getSettingValue("VideoPreview.AdvancedPreviewsDeadline")
                    // 仅在 VHS 存在时使用 /vhs/viewvideo，否则回退到 /view
                    this.videoEl.src = api.apiURL((useVHS ? '/vhs/viewvideo?' : '/view?') + new URLSearchParams(params));
                }
                this.videoEl.hidden = false;
                this.imgEl.hidden = true;
            } else if (params.format?.split('/')[0] == 'image'){
                // 是动画图像
                this.imgEl.src = api.apiURL('/view?' + new URLSearchParams(params));
                this.videoEl.hidden = true;
                this.imgEl.hidden = false;
            }
            
            delete previewNode.video_query
            const doQuery = async () => {
                if (!previewWidget?.value?.params?.filename) {
                    return
                }
                // VHS 查询仅在 VHS 存在时执行，避免 404
                if (!(app.widgets && app.widgets.VHSPATH)) return
                let qurl = api.apiURL('/vhs/queryvideo?' + new URLSearchParams(previewWidget.value.params))
                let query = undefined
                try {
                    let query_res = await fetch(qurl)
                    query = await query_res.json()
                } catch(e) {
                    return
                }
                previewNode.video_query = query
            }
            doQuery()
        }
        
        previewWidget.callback = previewWidget.updateSource
        previewWidget.parentEl.appendChild(previewWidget.videoEl)
        previewWidget.parentEl.appendChild(previewWidget.imgEl)
    });
}

/**
 * 为节点添加预览选项菜单
 * @param {Object} nodeType - 节点类型
 */
function addPreviewOptions(nodeType) {
    chainCallback(nodeType.prototype, "getExtraMenuOptions", function(_, options) {
        let optNew = []
        const previewWidget = this.widgets.find((w) => w.name === "videopreview");

        let url = null
        if (previewWidget.videoEl?.hidden == false && previewWidget.videoEl.src) {
            if (['input', 'output', 'temp'].includes(previewWidget.value.params.type)) {
                // 使用全质量视频
                url = api.apiURL('/view?' + new URLSearchParams(previewWidget.value.params));
                // 16位 png 的解决方案：只做第一帧
                url = url.replace('%2503d', '001')
            }
        } else if (previewWidget.imgEl?.hidden == false && previewWidget.imgEl.src) {
            url = previewWidget.imgEl.src;
            url = new URL(url);
        }
        
        if (this.video_query?.source) {
            let info_string = this.video_query.source.size.join('x') +
                              '@' + this.video_query.source.fps + 'fps ' +
                              this.video_query.source.frames + 'frames'
            optNew.push({content: info_string, disabled: true})
        }
        
        if (url) {
            optNew.push(
                {
                    content: "Open preview",
                    callback: () => {
                        window.open(url, "_blank")
                    },
                },
                {
                    content: "Save preview",
                    callback: () => {
                        const a = document.createElement("a");
                        a.href = url;
                        a.setAttribute("download", previewWidget.value.params.filename);
                        document.body.append(a);
                        a.click();
                        requestAnimationFrame(() => a.remove());
                    },
                }
            );
        }
        
        const PauseDesc = (previewWidget.value.paused ? "Resume" : "Pause") + " preview";
        if(previewWidget.videoEl.hidden == false) {
            optNew.push({content: PauseDesc, callback: () => {
                if(previewWidget.value.paused) {
                    previewWidget.videoEl?.play();
                } else {
                    previewWidget.videoEl?.pause();
                }
                previewWidget.value.paused = !previewWidget.value.paused;
            }});
        }
        
        const visDesc = (previewWidget.value.hidden ? "Show" : "Hide") + " preview";
        optNew.push({content: visDesc, callback: () => {
            if (!previewWidget.videoEl.hidden && !previewWidget.value.hidden) {
                previewWidget.videoEl.pause();
            } else if (previewWidget.value.hidden && !previewWidget.videoEl.hidden && !previewWidget.value.paused) {
                previewWidget.videoEl.play();
            }
            previewWidget.value.hidden = !previewWidget.value.hidden;
            previewWidget.parentEl.hidden = previewWidget.value.hidden;
            fitHeight(this);
        }});
        
        const muteDesc = (previewWidget.value.muted ? "Unmute" : "Mute") + " Preview"
        optNew.push({content: muteDesc, callback: () => {
            previewWidget.value.muted = !previewWidget.value.muted
        }})
        
        if(options.length > 0 && options[0] != null && optNew.length > 0) {
            optNew.push(null);
        }
        options.unshift(...optNew);
    });
}

/**
 * 为节点右键菜单添加“视频编辑”入口，打开模态编辑器进行矩形选择
 * @param {Object} nodeType - 节点类型
 */
function addVideoEditMenu(nodeType) {
    chainCallback(nodeType.prototype, "getExtraMenuOptions", function(_, options) {
        const previewWidget = this.widgets?.find((w) => w.name === "videopreview");
        if (!previewWidget) return;

        const openEditor = () => {
            // 构造视频地址（多重兜底）
            let src = previewWidget.videoEl?.currentSrc || previewWidget.videoEl?.src || null;
            const raw = Object.assign({}, previewWidget.value?.params || {});
            
            // 如果没有现成 src，尝试基于 params 组装（稳定使用 /view?）
            if (!src && Object.keys(raw).length > 0) {
                const params = Object.assign({}, raw);
                if (!params.type) params.type = 'input';
                const qs = new URLSearchParams(Object.assign({}, params, { timestamp: Date.now() }));
                src = api.apiURL('/view?' + qs);
            }
            
            // 仍然没有则尝试触发一次更新，让 videoEl 获得 src
            if (!src) {
                try { previewWidget.updateSource?.(); } catch (_) {}
                src = previewWidget.videoEl?.currentSrc || previewWidget.videoEl?.src || null;
            }
            
            // 继续兜底：直接读取节点的 video_path 组件，拼 /view? 请求
            if (!src) {
                const pathWidget = this.widgets?.find?.((w) => w.name === 'video_path');
                const val = pathWidget?.value;
                if (val && typeof val === 'string' && val.length > 0) {
                    let filename = val;
                    let subfolder = '';
                    const i = val.lastIndexOf('/');
                    if (i > 0) {
                        subfolder = val.slice(0, i);
                        filename = val.slice(i + 1);
                    }
                    const q = { filename, type: 'input', timestamp: Date.now() };
                    if (subfolder) q.subfolder = subfolder;
                    src = api.apiURL('/view?' + new URLSearchParams(q));
                }
            }
            
            if (!src) {
                alert('未找到可用的视频源，请先在节点中选择/预览视频');
                return;
            }

            // 背景遮罩
            const backdrop = document.createElement('div');
            backdrop.style.position = 'fixed';
            backdrop.style.left = '0';
            backdrop.style.top = '0';
            backdrop.style.width = '100vw';
            backdrop.style.height = '100vh';
            backdrop.style.background = 'rgba(0,0,0,0.6)';
            backdrop.style.zIndex = '100000';
            backdrop.style.display = 'flex';
            backdrop.style.alignItems = 'center';
            backdrop.style.justifyContent = 'center';

            // 编辑面板
            const panel = document.createElement('div');
            panel.style.position = 'relative';
            panel.style.background = '#111';
            panel.style.borderRadius = '8px';
            panel.style.boxShadow = '0 8px 32px rgba(0,0,0,0.4)';
            panel.style.width = '80vw';
            panel.style.maxWidth = '1280px';
            panel.style.padding = '12px 12px 16px 12px';
            panel.style.maxHeight = '90vh';
            panel.style.overflow = 'hidden';
            panel.style.display = 'flex';
            panel.style.flexDirection = 'column';

            // 标题栏
            const header = document.createElement('div');
            header.style.display = 'flex';
            header.style.alignItems = 'center';
            header.style.justifyContent = 'space-between';
            header.style.marginBottom = '8px';
            const title = document.createElement('div');
            title.textContent = '视频编辑 — 拖拽选择区域（双击清除）';
            title.style.color = '#fff';
            title.style.fontSize = '14px';
            title.style.fontWeight = '600';
            const btns = document.createElement('div');
            const closeBtn = document.createElement('button');
            closeBtn.textContent = '关闭';
            closeBtn.style.marginRight = '8px';
            const saveBtn = document.createElement('button');
            saveBtn.textContent = '保存';
            btns.appendChild(closeBtn);
            btns.appendChild(saveBtn);
            header.appendChild(title);
            header.appendChild(btns);

            // 视频容器
            const container = document.createElement('div');
            container.style.position = 'relative';
            container.style.width = '100%';
            container.style.userSelect = 'none';
            container.style.maxHeight = '70vh';
            container.style.overflow = 'hidden';

            const video = document.createElement('video');
            video.src = src;
            // 关闭浏览器原生控制条，避免其高度/区域干扰可点击与尺寸计算
            video.controls = false;
            video.autoplay = true;
            video.loop = false;
            video.muted = true;
            video.style.width = '100%';
            video.style.height = 'auto';
            video.style.maxHeight = '70vh';
            video.style.display = 'block';

            const overlay = document.createElement('div');
            overlay.style.position = 'absolute';
            overlay.style.left = '0';
            overlay.style.top = '0';
            // 初始设为 0，防止未对齐前覆盖容器
            overlay.style.width = '0px';
            overlay.style.height = '0px';
            overlay.style.pointerEvents = 'none';
            overlay.style.cursor = 'crosshair';
            overlay.style.zIndex = '10';

            const box = document.createElement('div');
            box.style.position = 'absolute';
            box.style.border = '2px solid #ff0000';
            box.style.backgroundColor = 'rgba(255,0,0,0.2)';
            box.style.display = 'none';
            box.style.pointerEvents = 'none';
            overlay.appendChild(box);

            let isDrawing = false;
            let overlayReady = false;
            // 显示坐标（overlay 内的像素）
            let startDX = 0; let startDY = 0; let currDX = 0; let currDY = 0;
            // 视频像素坐标（相对原视频分辨率）
            let startVX = 0; let startVY = 0; let currVX = 0; let currVY = 0;
            const getORect = () => overlay.getBoundingClientRect();
            const getVRect = () => video.getBoundingClientRect();
            const clamp = (v, min, max) => Math.max(min, Math.min(max, v));

            const onDown = (e) => {
                if (e.button !== 0) return;
                if (!overlayReady) return;
                const or = getORect();
                const vW = Math.max(1, video.videoWidth || or.width);
                const vH = Math.max(1, video.videoHeight || or.height);
                const vScaleX = vW / Math.max(1, or.width);
                const vScaleY = vH / Math.max(1, or.height);

                // 仅允许在视频内容矩形(overlay)内开始绘制
                if (e.clientX < or.left || e.clientX > or.right || e.clientY < or.top || e.clientY > or.bottom) {
                    return;
                }

                // 计算视频像素坐标（基于 overlay 内容矩形）
                startVX = clamp((e.clientX - or.left) * vScaleX, 0, vW);
                startVY = clamp((e.clientY - or.top)  * vScaleY, 0, vH);
                currVX = startVX; currVY = startVY;

                // 转换为覆盖层显示坐标
                startDX = (startVX / vW) * or.width;
                startDY = (startVY / vH) * or.height;
                currDX = startDX; currDY = startDY;

                isDrawing = true;
                box.style.display = 'block';
                box.style.left = startDX + 'px';
                box.style.top = startDY + 'px';
                box.style.width = '0px';
                box.style.height = '0px';
            };
            const onMove = (e) => {
                if (!isDrawing) return;
                const or = getORect();
                const vW = Math.max(1, video.videoWidth || or.width);
                const vH = Math.max(1, video.videoHeight || or.height);
                const vScaleX = vW / Math.max(1, or.width);
                const vScaleY = vH / Math.max(1, or.height);

                // 当前视频像素坐标（基于 overlay 内容矩形）
                const cx = clamp(e.clientX, or.left, or.right);
                const cy = clamp(e.clientY, or.top, or.bottom);
                currVX = clamp((cx - or.left) * vScaleX, 0, vW);
                currVY = clamp((cy - or.top)  * vScaleY, 0, vH);

                // 显示坐标
                currDX = (currVX / vW) * or.width;
                currDY = (currVY / vH) * or.height;

                const left = Math.min(startDX, currDX);
                const top = Math.min(startDY, currDY);
                const w = Math.abs(currDX - startDX);
                const h = Math.abs(currDY - startDY);
                box.style.left = left + 'px';
                box.style.top = top + 'px';
                box.style.width = w + 'px';
                box.style.height = h + 'px';
            };
            const onUp = (e) => { if (isDrawing && e.button === 0) isDrawing = false; };
            const onDbl = () => { box.style.display = 'none'; };

            overlay.addEventListener('mousedown', onDown);
            window.addEventListener('mousemove', onMove);
            window.addEventListener('mouseup', onUp);
            overlay.addEventListener('dblclick', onDbl);

            const cleanup = () => {
                overlay.removeEventListener('mousedown', onDown);
                window.removeEventListener('mousemove', onMove);
                window.removeEventListener('mouseup', onUp);
                overlay.removeEventListener('dblclick', onDbl);
                backdrop.remove();
            };

            closeBtn.onclick = cleanup;
            backdrop.onclick = (e) => { if (e.target === backdrop) cleanup(); };

            // 工具函数：设置/隐藏坐标widget并赋值
            const setHiddenCoord = (name, value) => {
                let w = this.widgets?.find?.((x) => x.name === name);
                if (!w) {
                    w = this.addWidget("number", name, 0, () => {}, { min: 0 });
                }
                if (w) {
                    w.options = w.options || {};
                    w.options.hidden = true;
                    w.computeSize = () => [0, -4];
                    w.value = value;
                    if (w.callback) w.callback(value);
                }
            };

            saveBtn.onclick = () => {
                // 直接使用视频像素坐标，避免二次换算误差
                const x1 = Math.round(Math.min(startVX, currVX));
                const y1 = Math.round(Math.min(startVY, currVY));
                const x2 = Math.round(Math.max(startVX, currVX));
                const y2 = Math.round(Math.max(startVY, currVY));

                // 写入隐藏坐标widget，保证序列化传给后端
                setHiddenCoord('crop_x1', x1);
                setHiddenCoord('crop_y1', y1);
                setHiddenCoord('crop_x2', x2);
                setHiddenCoord('crop_y2', y2);
                this.setDirtyCanvas?.(true);
                cleanup();
            };

            const syncOverlay = () => {
                const vr = video.getBoundingClientRect();
                const pr = container.getBoundingClientRect();
                const vW = Math.max(1, video.videoWidth || vr.width);
                const vH = Math.max(1, video.videoHeight || vr.height);
                const elemAR = vr.width / Math.max(1, vr.height);
                const videoAR = vW / vH;

                let contentW = vr.width;
                let contentH = vr.height;
                let offsetX = 0;
                let offsetY = 0;
                if (elemAR > videoAR) {
                    // 左右留黑边
                    contentH = vr.height;
                    contentW = contentH * videoAR;
                    offsetX = (vr.width - contentW) / 2;
                } else {
                    // 上下留黑边
                    contentW = vr.width;
                    contentH = contentW / videoAR;
                    offsetY = (vr.height - contentH) / 2;
                }

                overlay.style.width = contentW + 'px';
                overlay.style.height = contentH + 'px';
                overlay.style.left = (vr.left - pr.left + offsetX) + 'px';
                overlay.style.top = (vr.top - pr.top + offsetY) + 'px';
                overlayReady = contentW > 0 && contentH > 0;
                overlay.style.pointerEvents = overlayReady ? 'auto' : 'none';
            };

            video.addEventListener('loadedmetadata', () => requestAnimationFrame(syncOverlay));
            window.addEventListener('resize', syncOverlay);

            container.appendChild(video);
            container.appendChild(overlay);
            panel.appendChild(header);
            panel.appendChild(container);
            backdrop.appendChild(panel);
            document.body.appendChild(backdrop);
        };

        // 插入菜单项到最前
        options.unshift({ content: '视频编辑', callback: openEditor });
    });
}

/**
 * 在视频预览上启用选择框绘制（红色框），并把坐标写回节点属性
 * 坐标以视频实际分辨率为基准（videoWidth/videoHeight）
 * @param {Object} nodeType - 节点类型
 */
function addSelectionDrawing(nodeType) {
    chainCallback(nodeType.prototype, "onNodeCreated", function() {
        const node = this;
        const previewWidget = node.widgets?.find((w) => w.name === "videopreview");
        if (!previewWidget || !previewWidget.parentEl) return;

        // 查找/创建四个隐藏坐标输入widget（如果存在则复用）
        const getOrCreateHidden = (name) => {
            let w = node.widgets?.find((x) => x.name === name);
            if (!w) {
                w = node.addWidget("number", name, 0, () => {}, { min: 0 });
            }
            // 隐藏该widget在UI
            if (w) {
                w.options = w.options || {};
                w.options.hidden = true;
                w.computeSize = () => [0, -4];
            }
            return w;
        };
        const wx1 = getOrCreateHidden('crop_x1');
        const wy1 = getOrCreateHidden('crop_y1');
        const wx2 = getOrCreateHidden('crop_x2');
        const wy2 = getOrCreateHidden('crop_y2');

        // 防重复初始化
        if (previewWidget.__selection_inited__) return;
        previewWidget.__selection_inited__ = true;

        // 覆盖层
        const overlay = document.createElement("div");
        overlay.style.position = "absolute";
        overlay.style.left = "0";
        overlay.style.top = "0";
        overlay.style.width = "100%";
        overlay.style.height = "100%";
        overlay.style.pointerEvents = "auto";
        overlay.style.zIndex = "1000";
        overlay.style.background = "transparent";
        overlay.style.cursor = "crosshair";
        previewWidget.parentEl.appendChild(overlay);

        // 选择框
        const box = document.createElement("div");
        box.style.position = "absolute";
        box.style.border = "2px solid #ff0000";
        box.style.backgroundColor = "rgba(255,0,0,0.2)";
        box.style.display = "none";
        box.style.pointerEvents = "none";
        overlay.appendChild(box);

        let isDrawing = false;
        let startX = 0, startY = 0;

        	const getClientRect = () => overlay.getBoundingClientRect();

        	// 同步 overlay 到视频内容区域（考虑缩放/letterbox）
        	const syncInlineOverlay = () => {
        		const v = previewWidget.videoEl;
        		if (!v) return;
        		const vr = v.getBoundingClientRect();
        		const pr = previewWidget.parentEl.getBoundingClientRect();
        		overlay.style.width = vr.width + 'px';
        		overlay.style.height = vr.height + 'px';
        		overlay.style.left = (vr.left - pr.left) + 'px';
        		overlay.style.top = (vr.top - pr.top) + 'px';
        	};
        
        	// 在元数据就绪和窗口尺寸变化时保持同步
        	previewWidget.videoEl?.addEventListener('loadedmetadata', () => requestAnimationFrame(syncInlineOverlay));
        	window.addEventListener('resize', syncInlineOverlay);
		// 监听视频元素尺寸变化（节点缩放/面板变化）
		try { new ResizeObserver(syncInlineOverlay).observe(previewWidget.videoEl); } catch(_) {}
		// 初次挂载后立即同步一次，避免 100% 覆盖黑边
		requestAnimationFrame(syncInlineOverlay);

		const onDown = (e) => {
            if (e.button !== 0) return; // 仅左键
			// 开始前强制同步，确保 overlay 已贴合视频矩形
			syncInlineOverlay();
			const rect = getClientRect();
			// 限制只能在视频矩形内按下（黑边不可用）
			if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) {
				return;
			}
            startX = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
            startY = Math.max(0, Math.min(e.clientY - rect.top, rect.height));
            isDrawing = true;
            box.style.display = "block";
            box.style.left = startX + "px";
            box.style.top = startY + "px";
            box.style.width = "0px";
            box.style.height = "0px";
        };

        const onMove = (e) => {
            if (!isDrawing) return;
            const rect = getClientRect();
            const cx = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
            const cy = Math.max(0, Math.min(e.clientY - rect.top, rect.height));
            const left = Math.min(startX, cx);
            const top = Math.min(startY, cy);
            const w = Math.abs(cx - startX);
            const h = Math.abs(cy - startY);
            box.style.left = left + "px";
            box.style.top = top + "px";
            box.style.width = w + "px";
            box.style.height = h + "px";
        };

        const onUp = (e) => {
            if (!isDrawing || e.button !== 0) return;
            isDrawing = false;
            const rect = getClientRect();
            const endX = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
            const endY = Math.max(0, Math.min(e.clientY - rect.top, rect.height));
            const left = Math.min(startX, endX);
            const top = Math.min(startY, endY);
            const right = Math.max(startX, endX);
            const bottom = Math.max(startY, endY);

            // 映射到视频实际分辨率（严格以 overlay 可见区域为显示尺寸）
            const v = previewWidget.videoEl;
            const videoW = Math.max(1, v?.videoWidth || rect.width);
            const videoH = Math.max(1, v?.videoHeight || rect.height);
            const scaleX = videoW / Math.max(1, rect.width);
            const scaleY = videoH / Math.max(1, rect.height);

            const x1 = Math.round(left * scaleX);
            const y1 = Math.round(top * scaleY);
            const x2 = Math.round(right * scaleX);
            const y2 = Math.round(bottom * scaleY);

            // 写回隐藏widget，触发序列化传给后端
            const setVal = (w, v) => { if (w) { w.value = v; if (w.callback) w.callback(v); } };
            setVal(wx1, x1); setVal(wy1, y1); setVal(wx2, x2); setVal(wy2, y2);
            node.setDirtyCanvas?.(true);
        };

        const onDblClick = () => {
            box.style.display = "none";
            const setZero = (w) => { if (w) { w.value = 0; if (w.callback) w.callback(0); } };
            setZero(wx1); setZero(wy1); setZero(wx2); setZero(wy2);
            node.setDirtyCanvas?.(true);
        };

        overlay.addEventListener("mousedown", onDown);
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp);
        overlay.addEventListener("dblclick", onDblClick);

        // 清理
        chainCallback(node, "onRemoved", () => {
            overlay.removeEventListener("mousedown", onDown);
            window.removeEventListener("mousemove", onMove);
            window.removeEventListener("mouseup", onUp);
            overlay.removeEventListener("dblclick", onDblClick);
        });
    });
}

// 注册扩展
app.registerExtension({
    name: "VideoPreview.Upload",
    settings: [
      {
        id: 'VideoPreview.AdvancedPreviews',
        category: ['Video Tools', 'Preview Display', 'Advanced Options'],
        name: 'Advanced Previews',
        tooltip: 'Automatically transcode previews on request. Required for advanced functionality',
        type: 'combo',
        options: ['Never', 'Always', 'Input Only'],
        defaultValue: 'Input Only',
      },
      {
        id: 'VideoPreview.AdvancedPreviewsMinWidth',
        category: ['Video Tools', 'Preview Display', 'Size Settings'],
        name: 'Minimum preview width',
        tooltip: 'Advanced previews have their resolution downscaled to the node size for performance.',
        type: 'number',
        attrs: {
          min: 0,
          step: 1,
          max: 3840,
        },
        defaultValue: 0,
      },
      {
        id: 'VideoPreview.AdvancedPreviewsDeadline',
        category: ['Video Tools', 'Preview Display', 'Performance'],
        name: 'Deadline',
        tooltip: 'Determines how much time can be spent when encoding advanced previews.',
        type: 'combo',
        options: ['realtime', 'good'],
        defaultValue: 'realtime',
      },
      {
        id: 'VideoPreview.AdvancedPreviewsDefaultMute',
        category: ['Video Tools', 'Preview Display', 'Audio Settings'],
        name: 'Mute videos by default',
        type: 'boolean',
        defaultValue: false,
      },
    ],

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 为视频画幅裁切节点添加预览功能
        if (nodeData?.name === "VideoCropNode") {
            // 添加视频预览功能
            addVideoPreview(nodeType, true);
            
            // 为输入文件夹添加文件夹上传功能
            addUploadWidget(nodeType, nodeData, "input_folder", "folder");
        }
        
        // 为视频预览节点添加预览功能
        if (nodeData?.name === "VideoPreviewNode") {
            // 添加视频预览功能
            addVideoPreview(nodeType, true);

            // 右键“视频编辑”面板
            addVideoEditMenu(nodeType);

            // 上传按钮（选择本地视频并回填）
            addUploadWidget(nodeType, nodeData, "video_path", "video");

            // 绑定路径选择回调，联动预览（路径可来自对话框/手动输入/上传回填）
            chainCallback(nodeType.prototype, "onNodeCreated", function() {
                const pathWidget = this.widgets?.find?.((w) => w.name === "video_path");
                if (!pathWidget) return;
                // 过滤扩展名（用于 VHS 风格路径选择器兼容）
                pathWidget.options = pathWidget.options || {};
                pathWidget.options.vhs_path_extensions = ["mp4","mov","mkv","webm","avi","wmv","flv","m4v","gif"];
                chainCallback(pathWidget, "callback", (value) => {
                    if (!value) return;
                    const dot = value.lastIndexOf(".");
                    const ext = dot >= 0 ? value.slice(dot+1).toLowerCase() : "mp4";
                    let format = ["gif","webp","avif"].includes(ext) ? "image" : "video";
                    format += "/" + ext;
                    // 推断 /view 标准参数：type(input/output/temp) + subfolder + filename
                    const inferParams = (fullPath) => {
                        let p = fullPath.replace(/\\/g, '/');
                        const parts = p.split('/').filter(Boolean);
                        const idxInput = parts.lastIndexOf('input');
                        const idxOutput = parts.lastIndexOf('output');
                        const idxTemp = parts.lastIndexOf('temp');
                        let type = 'input';
                        let subfolder = '';
                        let filename = parts.length ? parts[parts.length - 1] : fullPath;
                        if (idxInput >= 0) {
                            type = 'input';
                            subfolder = parts.slice(idxInput + 1, parts.length - 1).join('/');
                        } else if (idxOutput >= 0) {
                            type = 'output';
                            subfolder = parts.slice(idxOutput + 1, parts.length - 1).join('/');
                        } else if (idxTemp >= 0) {
                            type = 'temp';
                            subfolder = parts.slice(idxTemp + 1, parts.length - 1).join('/');
                        } else {
                            // 无法匹配受控根目录，尽量按相对路径处理
                            const i = p.lastIndexOf('/');
                            if (i > 0) {
                                subfolder = p.slice(0, i);
                                filename = p.slice(i + 1);
                            }
                        }
                        const out = { filename, type };
                        if (subfolder) out.subfolder = subfolder;
                        return out;
                    }
                    const base = inferParams(value);
                    const params = { ...base, format };
                    this.updateParameters(params, true);
                });
            });
        }
    },
    
    async init() {
        if (app.ui.settings.getSettingValue("VideoPreview.AdvancedPreviews") == true) {
            app.ui.settings.setSettingValue("VideoPreview.AdvancedPreviews", 'Always')
        }
        if (app.ui.settings.getSettingValue("VideoPreview.AdvancedPreviews") == false) {
            app.ui.settings.setSettingValue("VideoPreview.AdvancedPreviews", 'Never')
        }
    },
});

// 导出函数供外部使用
window.VHSCore = {
    addVideoPreview,
    addUploadWidget,
    addPreviewOptions,
    chainCallback,
    fitHeight,
    allowDragFromWidget,
    uploadFile
};

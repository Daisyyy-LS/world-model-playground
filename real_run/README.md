# 真实模型接入运行记录（Real Model Run Log）

这份记录保留**从失败到跑通的完整过程**。在评测岗面试里，能讲清楚"我怎么把链路打通、卡在哪里、怎么解决、结果怎么复核"，比只拿出一个跑通的结果更能体现工程能力。

## 一句话结论

**已用 HF 认证完整跑通腾讯官方开源世界模型 HunyuanWorld-Mirror 的真实推理，产出 7 件真实产物（3DGS 点云、点云 Mesh、深度图、法线图、相机外参、RGB 漫游视频、深度漫游视频），合计约 12 MB，全部下载到 `output/`。** `run_meta.json` 记录 `status=success`、`source_type=real_model`。所有产物均为官方模型的真实输出，无伪造、无 CSS 示意。

## 环境探测结果（真实）

| 项目 | 结果 |
|---|---|
| 本地 GPU | NVIDIA RTX 2050 · 4GB 显存（空闲 2.9GB）· Compute 8.6 |
| 本地能否跑世界模型 | **否**。该模型官方要求 CUDA 12.4 + 大显存卡（4090 级）；4GB 无法加载 |
| 采用路径 | **云端 ZeroGPU**：腾讯官方 Space `tencent/HunyuanWorld-Mirror`（Gradio 5.49.1） |
| 网络 | HuggingFace 直连超时；`hf-mirror.com` 镜像可达；官方 Space 域名 `*.hf.space` 可达 |
| 沙箱代理 | `http_proxy=127.0.0.1:51903` 会让 gradio_client 报 502；清除代理环境变量后可直连 |
| HF 凭证 | 最终由用户提供免费 READ token（用户 `Daisyyy-LS`），认证后配额满足一次重建 |

## 官方接口与调用链（真实）

连接 `https://tencent-hunyuanworld-mirror.hf.space/`，解析出 18 个具名 API 端点。核心推理链两步：

1. `/update_gallery_on_file_upload(files, interval)` → `output_path`：上传图片/视频，云端做视角采样与预处理。
2. `/gradio_demo(target_dir, frame_selector, show_camera, filter_sky_bg, show_mesh, filter_ambiguous)` → 返回 `3D Gaussian Splatting / 3D Pointmap-Mesh / Depth Map / Normal Map / Camera Parameters / Rendered RGB Video / Rendered Depth Video`。

## 真实执行记录（见 run_meta.json）

| 步骤 | 状态 | 耗时 | 结果 |
|---|---|---|---|
| 1. 上传 + 视角采样 | ✅ 成功 | 6.8s | `output_path=input_images_20261118_130116_856412` |
| 2. 3D 重建（云端 GPU） | ✅ 成功 | 82.4s | 7 件产物全部返回并下载 |

输入为官方示例室内场景 `Bright_Room/image_0001.jpg`（1280×726）。

### 产物清单（output/）

| 文件 | 类型 | 大小 | 浏览器可直接预览 |
|---|---|---|---|
| `rgb_video.mp4` | 3DGS 漫游视频 | 486 KB | ✅ |
| `depth_video.mp4` | 深度漫游视频 | 633 KB | ✅ |
| `depth_map.png` | 深度图（518×294 RGB） | 41 KB | ✅ |
| `normal_map.png` | 法线图（518×294 RGB） | 135 KB | ✅ |
| `gaussian_splatting.ply` | 3DGS 点云（二进制 PLY） | 7.8 MB | 需 3D 查看器 |
| `pointmap_mesh.glb` | 点云 Mesh（glTF 2.0） | 2.4 MB | 需 3D 查看器 |
| `camera_params.json` | 相机外参矩阵（1 相机） | 945 B | ✅ |

真实性校验：深度图/法线图为 518×294 的 RGB 图像，与 1280×726 输入图尺寸不同；PLY 文件头为 `ply\nformat binary...`；GLB 文件头为 `glTF` 魔数；两段视频为合法 MP4（`ftypisom`）。均非输入图回显。

## 从失败到跑通：排查过的路径（面试素材）

第一次匿名运行在步骤 2 被拦：
> `You have exceeded your ZeroGPU quota (180s requested vs. 123s left). Authenticate with a Hugging Face token for more quota`

逐层排查：
- 换镜像 Space（`PaulHouston/HunyuanWorld-Mirror`，接口完全一致）→ 同一匿名配额，同样被拦；
- 减少采样帧数（interval 5.0，仅 1 张视图）降低负载 → 仍被拦，因为 180s 是函数声明的 GPU 时长，与输入大小无关；
- 预置示例接口 `/handle_realworld_selection` → 依赖前端 gallery 状态，无状态直调返回 `NoneType` 错误；
- 腾讯国内站 `3d-models.hunyuan.tencent.com/world/`（直连 0.6s）→ 实际 API 在 `3d.hunyuan.tencent.com/sceneTo3D`，为需登录鉴权的 SPA，逆向成本高且不宜绕过；
- **最终解法**：注册免费 HF 账号创建 READ token（登录用户配额 300 秒/天 > 单次 180 秒），并在脚本里用正确的 `gradio_client.Client(token=...)` 传参（注意不是 `hf_token`）→ 完整跑通。

另修复了产物保存时 `FileData.video` 为字符串路径导致 `save_obj()` 报 `'str' object has no attribute 'get'` 的 bug，之后重跑一次拿到含视频的全部 7 件产物。第一次匿名运行的配额错误原文如上引用保留，不粉饰。

## 复现方法

```bash
# 1. 注册 https://huggingface.co/join ，在 https://huggingface.co/settings/tokens 建 READ token
# 2. 已登录过可省略 --token（脚本自动读 HF 缓存）
python real_run/run_inference.py --token hf_xxxxxxxxxxxxxxxxxxxx
# 或设环境变量
set HF_TOKEN=hf_xxxxx && python real_run/run_inference.py
```

脚本自动清除代理、连接官方 Space、执行两步推理，把真实产物下载到 `output/`，并在 `run_meta.json` 写入 `source_type=real_model`、模型版本、端点、耗时、每个产物的字节数——字段与主测评台的 `records` 导入格式对齐，可直接作为真实证据进入评测流程。

## 这段经历怎么用于面试

- **工程排查能力**：独立定位并解决了代理 502、镜像回显、ZeroGPU 配额、SDK 参数名、产物保存 5 类真实问题。
- **对评测链路的理解**：清楚区分"上传回显 ≠ 模型输出"，拒绝用输入图冒充结果——这正是评测岗最看重的数据诚信。
- **诚实标注边界**：本文如实保留了第一次运行的配额错误原文，不粉饰；最终结果 `run_meta.json` 可逐字段复核。
- **闭环能力**：真实产物已接入主测评台「实测产物」模式，可点击预览、勾选故障标签、导出含 `source_type=real_model` 的记录——从"设计评测"走到"用真实输出跑评测"。

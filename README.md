# DOLPHIN

This is a temporal graph training system.
## Project Structure

The repository is organized as follows:

```
/home/dzz/tgl/
├── a-tgl/                      # Official TGL implementation **这个版本是tgl原版用于测时间、内存、显存的**
├── a-tgl-AP/                   # Official TGL implementation with AP Testing**这里是测试准确率**
├── a-tgl-s/                    # Official TGL implementation with Runtime and Main Memory Cost Testing **采样消融是找tgl原版采样bug的**
├── a-tgl-test/                 # **基本没用到 试着跑了一下tgl**
├── b-tgl/                      # $\alpha$ TGL implementation
├── dag                         # **没有用**
├── DOLPHIN/                    # $\beta$ DOLPHIN implementation
├── ETC/                        # Experimental ETC (VLDB 2024) implementations
├── alpha-exp/                  # Experimental folder for $\alpha$ Version implementations**最近的实验结果**
├── simple/                     # Experimental SIMPLE (SIGMOD 2024) implementations **一个是tgl原版的采样 tgl采样是他写的cpu多线程采样 是有bug的（不确定 只是觉得有bug）**
├── simple-s/                   # ** 一个是修改后的采样算法（也是cpp实现的） **
├── test/                       # Test folder for various implementations, including xxxx**没有作用 最开始测试采样**
├── BITCOIN.json                # Bitcoin dataset configuration**论文画图的**
├── bucket_cache_node_map.pt    # Bucket cache node mapping**实验中间产物**
├── dataset.json                 # Some Information about datasets 各数据集的边数、点数、特征维度、存储大小等信息**
├── GDELT.json                   # GDELT dataset configuration**论文画图的**
├── STACK.json                   # STACK dataset configuration**论文画图的**
├── watcher.sh                   # **Watcher script**???**没有作用 定位哪些进程 / 用户在消耗大量 inotify 资源，帮助诊断这类问题。例如：如果 VS Code 打开了大量文件，可能会耗尽 inotify watches，导致无法监控新文件，此时用脚本可快速发现是 VS Code 程占用了过多资源。**
```

Each folder contains relevant files to facilitate development, experimentation, and deployment.
## Features

- Feature 1
- Feature 2
- Feature 3

## Getting Started

### Prerequisites

- List any requirements or dependencies

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/yourproject.git
cd yourproject

# Install dependencies
# (Add installation commands here)
```

## Usage

```bash
# Example usage
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the [LICENSE](LICENSE) file.
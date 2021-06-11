### 配置conda环境
conda create -n py36gpu python=3.6
pip install --upgrade pip
### 配置torch环境(根据实际cuda版本进行版本选择)
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html -i https://pypi.tuna.tsinghua.edu.cn/simple
### 安装其他
pip install scipy scikit-learn==0.19 requests tqdm pyyaml matplotlib pandas seaborn opencv-python paho-mqtt pymysql

### 模型文件下载
链接: https://pan.baidu.com/s/1yOEEECw_XMtDQN3JqFAUzQ 提取码: j5ay 复制这段内容后打开百度网盘手机App，操作更方便哦


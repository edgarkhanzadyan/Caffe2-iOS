#echo "[Step1] Installing dependencies"
#brew install automake libtool

echo "[Step1] git cloning Caffe2"
git clone --recursive https://github.com/caffe2/caffe2.git

echo "[Step2] checkout to a stable version tags/v0.7.0-163-gebc17cc8"
cd caffe2 && git reset --hard 01827c153d

echo "[Step3] update submodules"
git submodule update

echo "[Step3] build ios"
./scripts/build_ios.sh

echo "[Tada!] Done"
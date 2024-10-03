cd root
pip install py-spy
apt-get -y install sudo curl
curl https://getcroc.schollz.com | bash
echo 'from .maploc.demo import Demo;demo=Demo()' > src/tst.py
sudo py-spy record -f speedscope -o perf.json -- python -m src.tst
sudo py-spy record -f speedscope -o perf2.json -- python -m src.tst
croc send perf.json perf2.json
clean:
	rm -r runs traced_model.pt
local_demo:
	python streamer_demo.py --video sample/test_video.mp4 --loop --host 127.0.0.1 --port 8002
TAG 			:= latest	
USER 			:= mlexchange1
PROJECT			:= dlsia

IMG_WEB_SVC    		:= ${USER}/${PROJECT}:${TAG}
IMG_WEB_SVC_JYP    	:= ${USER}/${PROJECT_JYP}:${TAG}
ID_USER			:= ${shell id -u}
ID_GROUP			:= ${shell id -g}

.PHONY:

test:
	echo ${IMG_WEB_SVC}
	echo ${TAG}
	echo ${PROJECT}
	echo ${PROJECT}:${TAG}
	echo ${ID_USER}

build_docker: 
	docker build -t ${IMG_WEB_SVC} -f ./docker/Dockerfile .

run_docker:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it -v ${PWD}:/app/work/local -p 8888:8888 ${IMG_WEB_SVC}

train_msdnet_maxdil:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it -v ${PWD}:/app/work/ ${IMG_WEB_SVC} python3 src/train.py data/mask data/images/train data/output '{"model": "MSDNet", "num_epochs": 10, "optimizer": "Adam", "criterion": "CrossEntropyLoss", "learning_rate": 0.01, "num_layers": 10, "custom_dilation": false, "max_dilation": 5}'

train_msdnet_customdil:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it -v ${PWD}:/app/work/ ${IMG_WEB_SVC} python3 src/train.py data/mask data/images/train data/output '{"model": "MSDNet", "num_epochs": 10, "optimizer": "Adam", "criterion": "CrossEntropyLoss", "learning_rate": 0.01, "num_layers": 10, "custom_dilation": true, "dilation_array": [1,2,5]}'

train_tunet:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it -v ${PWD}:/app/work/ ${IMG_WEB_SVC} python3 src/train.py data/mask data/images/train data/output '{"model": "TUNet", "num_epochs": 10, "optimizer": "Adam", "criterion": "CrossEntropyLoss", "learning_rate": 0.01, "depth": 4, "base_channels": 16, "growth_rate": 2, "hidden_rate": 1}'

train_tunet3plus:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it -v ${PWD}:/app/work/ ${IMG_WEB_SVC} python3 src/train.py data/mask data/images/train data/output '{"model": "TUNet3+", "num_epochs": 10, "optimizer": "Adam", "criterion": "CrossEntropyLoss", "learning_rate": 0.01, "depth": 4, "base_channels": 16, "carryover_channels": 5, "growth_rate": 2, "hidden_rate": 1}'

test_segment:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --memory-swap -1 -it -v ${PWD}:/app/work/ ${IMG_WEB_SVC} python3 src/segment.py data/images/test/segment_series.tif data/output/state_dict_net.pt data/output '{"show_progress": 1}'

clean: 
	find -name "*~" -delete
	-rm .python_history
	-rm -rf .config
	-rm -rf .cache

push_docker:
	docker push ${IMG_WEB_SVC}

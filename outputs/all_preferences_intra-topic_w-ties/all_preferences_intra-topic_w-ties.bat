@echo Running Better Rewards all_preferences_intra-topic_w-ties

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.3
set model_type=linear
set device=gpu
set seed=1
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL

@echo Finished 0.3 linear 1

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.3
set model_type=linear
set device=gpu
set seed=2
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL

@echo Finished 0.3 linear 2

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.3
set model_type=linear
set device=gpu
set seed=3
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL

@echo Finished 0.3 linear 3

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.3
set model_type=deep
set device=gpu
set seed=1
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL

@echo Finished 0.3 deep 1

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.3
set model_type=deep
set device=gpu
set seed=2
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL


@echo Finished 0.3 deep 2

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.3
set model_type=deep
set device=gpu
set seed=3
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL


@echo Finished 0.3 deep 3

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.03
set model_type=linear
set device=gpu
set seed=1
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL

@echo Finished 0.03 linear 1

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.03
set model_type=linear
set device=gpu
set seed=2
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL

@echo Finished 0.03 linear 2

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.03
set model_type=linear
set device=gpu
set seed=3
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL

@echo Finished 0.03 linear 3

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.03
set model_type=deep
set device=gpu
set seed=1
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL

@echo Finished 0.03 deep 1

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.03
set model_type=deep
set device=gpu
set seed=2
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL


@echo Finished 0.03 deep 2

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.03
set model_type=deep
set device=gpu
set seed=3
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL


@echo Finished 0.03 deep 3

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.003
set model_type=linear
set device=gpu
set seed=1
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL

@echo Finished 0.003 linear 1

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.003
set model_type=linear
set device=gpu
set seed=2
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL

@echo Finished 0.003 linear 2

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.003
set model_type=linear
set device=gpu
set seed=3
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL

@echo Finished 0.003 linear 3

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.003
set model_type=deep
set device=gpu
set seed=1
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL

@echo Finished 0.003 deep 1

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.003
set model_type=deep
set device=gpu
set seed=2
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL


@echo Finished 0.003 deep 2

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.003
set model_type=deep
set device=gpu
set seed=3
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL


@echo Finished 0.003 deep 3

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.0003
set model_type=linear
set device=gpu
set seed=1
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL

@echo Finished 0.0003 linear 1

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.0003
set model_type=linear
set device=gpu
set seed=2
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL

@echo Finished 0.0003 linear 2

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.0003
set model_type=linear
set device=gpu
set seed=3
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL

@echo Finished 0.0003 linear 3

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.0003
set model_type=deep
set device=gpu
set seed=1
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL

@echo Finished 0.0003 deep 1

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.0003
set model_type=deep
set device=gpu
set seed=2
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL


@echo Finished 0.0003 deep 2

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.0003
set model_type=deep
set device=gpu
set seed=3
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL


@echo Finished 0.0003 deep 3


SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.00003
set model_type=linear
set device=gpu
set seed=1
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL

@echo Finished 0.00003 linear 1

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.00003
set model_type=linear
set device=gpu
set seed=2
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL

@echo Finished 0.00003 linear 2

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.00003
set model_type=linear
set device=gpu
set seed=3
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL

@echo Finished 0.00003 linear 3

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.00003
set model_type=deep
set device=gpu
set seed=1
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL

@echo Finished 0.00003 deep 1

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.00003
set model_type=deep
set device=gpu
set seed=2
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL


@echo Finished 0.00003 deep 2

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.00003
set model_type=deep
set device=gpu
set seed=3
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL


@echo Finished 0.00003 deep 3

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.000003
set model_type=linear
set device=gpu
set seed=1
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL

@echo Finished 0.000003 linear 1

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.000003
set model_type=linear
set device=gpu
set seed=2
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL

@echo Finished 0.000003 linear 2

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.000003
set model_type=linear
set device=gpu
set seed=3
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL

@echo Finished 0.000003 linear 3

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.000003
set model_type=deep
set device=gpu
set seed=1
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL

@echo Finished 0.000003 deep 1

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.000003
set model_type=deep
set device=gpu
set seed=2
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL


@echo Finished 0.000003 deep 2

SETLOCAL

set epoch_num=50
set batch_size=32
set train_type=linear
set train_percent=0.64
set dev_percent=0.16
set learn_rate=0.000003
set model_type=deep
set device=gpu
set seed=3
set file_name=all_preferences_intra-topic_w-ties.csv

call F:\Python\python.exe F:\PythonProjects\summary-reward-no-reference-master\step2_train_rewarder.py --epoch_num %epoch_num% --batch_size %batch_size% --train_type %train_type% --train_percent %train_percent%  --dev_percent %dev_percent% --learn_rate %learn_rate% --model_type %model_type% --device %device% --seed %seed% --file_name %file_name%

ENDLOCAL


@echo Finished 0.000003 deep 3


@echo Finished all

pause
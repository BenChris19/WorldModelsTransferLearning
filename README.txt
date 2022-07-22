***************************How to execute the code for World Models************************

Note: I did not include any of the collected data to run this code, otherwise the folder would be too big to submit

FIRST STEP:
	·Run the randomCar, randomLunar and randomBipedal code in order to collect 10.000 images from each OpenAi task.
SECOND STEP:
	·Run the TrainVision or TransferTrainVision (Depending on the training method you want to use) to tran the CVAE
	 on the 10.000 images. This can take up to 10-14 hours to execute.
THIRD STEP:
	·Run the series, seriesLunar and seriresBipedal code to preprocess the images
FOURTH STEP:
	.Run the trainMemory, trainMemoryLunar and trainMemoryBipedal, in order to train the M model.
	 This may take up 10-20 minutes to execute.
FIFTH STEP:
	.Run any of the TrainController files in order to train the controller on the World Models. Note, that for each
	 generation to finish executing, this may take up between 40 minutes and 1 hour. If you want to achieve optimal
	 results for the CarRacing-v0 task, you must wait a week for the code to finish executing.

*****************************************************************************************************
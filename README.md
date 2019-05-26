## <b> Task explanation:</b>

> You work as a data scientists at one of the top universities in the USA. One day a rector of the university comes to you with a task. She wants you to investigate the university’s admittance criteria and to create an engine that would recommend candidates with the highest probability of graduating. The university wants to maximise the graduate rates by enrolling only such students, because it is beneficial for both students and the university. Students get their degree, which makes it easier for them to start their career, and the university earns a tuition for the whole length of a program, as dropouts are minimised.

>You start your investigation by gathering two sets of data:

		● score_board.csv - contains candidates data with information on whether they were admitted or not;
		● graduates.csv - consists only of students data (so only those that started their education the university are included) with information on whether they graduated or not.
>Columns definitions:
	
		● id - candidate id;
		● year - recruitment year;
		● gpa - Grade Point Average;
		● maths_exam, physics_exam, cs_exam, art_exam, language_exam - various exam results;
		● social_activity - score for the social activities (volunteering etc), values from 1-5, 1 means least active, 5 most active;
		● essay_score, interview_score - scores for an entry essay and an interview;
		● score - total score calculated to rank candidates;
		● accepted - whether a given candidate was high enough in a ranking to be accepted and whether they decided to use this opportunity;
		● graduated - whether a student graduated;

## <b> Solution: </b>

 1. Firstly I discovered the data and prepared it for machine learning. The steps I made:

	 - combine graduates.csv and score_board.csv into one file.csv
	 - convert categorical columns into numeric (for example: FALSE = 0, TRUE = 1)
	 - round the floating points values to three decimals for improving learning accuracy

2. Preparing data for training the model:

	 - graduated column provides three informations: TRUE, FALSE, NO_INFO. We do not need our model to train on those three possible outcomes because we want at the output information whether student will graduate or not. Based on this assumption I made score_board_train.csv which contains students with graduated label = TRUE or FALSE. This dramatically decrease possible data to train from almost 50 000 records to 7 000. 
 
	 - next step was to delete unwanted columns for training such as: 
 'graduated' (this is output information that we want to predict), 
 'id' (does not influence student's graduation), 
 'accepted' (we already sorted that information), 
 'year' (does not influence student's graduation).
 
	 - division of previously prepared dataset into 'train' and 'test' set with ratio of 85% - 15% respectively (this results with the best model accuracy).
 3. Model training and evaluating its accuracy.
	 - tested six different classifiers and obtained such results:
	 <br>
	 | Gaussian   |  = 64.9786% 
	 <br>
	 | KNeighbors | = 69.0442% (used in my app, because of the highest accuracy)
	 <br>
	 | LinearSVC | = 68.1169%
	 <br>
	 | RandomForrest | = 67.1184%
	 <br>
	 | Multi-Layer Perceptron | = 32.2396%
	 <br>
	 | AdaBoost | = 68.4736%

4. App preparation in microframework - Flask and deployment on Google Cloud Platform.
		
	- deployed app can be seen at: [pearson-classifier.appspot.com](http://pearson-classifier.appspot.com/)
## How to deploy app on local host:
1.  In prepared virtual environment run following command: 
    <br>git clone  ([https://github.com/AniQl/predicting_student_graduation](https://github.com/AniQl/predicting_student_graduation).git)
    
2.  Install all dependencies by:
    <br>pip install -r requirements.txt
    
3.  Run the server from /predicting_student_graduation folder
    <br>python3 flask_app.py
    
4.  Enter and test:  
    [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
	
## Future work (things I did not manage to do in time):

 1. Unfortunetly I did not prepared the jupyter notebook report due to my long unavaibility. I wanted to present all data distribution and dependecies between different features using plots.
 2. Improve front-end application.
 3. More thorough explanation of my methods and way of thinking.
 4. Collect more data by using model trained on (7000k records) to predict rest (40 000 records) and then train new model based on previously predicted records. 


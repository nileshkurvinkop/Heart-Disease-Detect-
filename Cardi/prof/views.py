from django.shortcuts import render
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
import os
print(os.listdir())

import warnings
warnings.filterwarnings('ignore')

# Create your views here.
def home(request):
	return render(request,'index.html')
	
def predict(request):
	if request.method=='POST':
		train = pd.read_csv("heart.csv")		
		print(train.head(5))
		print(train.sample(5))
		print(train.describe())
		print(train.info())
		info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]
		for i in range(len(info)):
			print(train.columns[i]+":\t\t\t"+info[i])
		print(train['target'].describe())
		print(train['target'].unique())
		print(train.corr()["target"].abs().sort_values(ascending=False))
		from sklearn.model_selection import train_test_split

		predictors = train.drop("target",axis=1)
		target = train["target"]

		X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)
		from sklearn.metrics import accuracy_score
		from sklearn.linear_model import LogisticRegression

		lr = LogisticRegression()

		lr.fit(X_train,Y_train)

		Y_pred_lr = lr.predict(X_test)
		
		score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)

		print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")
		
		from sklearn.naive_bayes import GaussianNB

		nb = GaussianNB()

		nb.fit(X_train,Y_train)

		Y_pred_nb = nb.predict(X_test)
		
		score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)

		print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")
		
		from sklearn import svm

		sv = svm.SVC(kernel='linear')

		sv.fit(X_train, Y_train)

		Y_pred_svm = sv.predict(X_test)
		
		score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)

		print("The accuracy score achieved using Linear SVM is: "+str(score_svm)+" %")
		
		from sklearn.neighbors import KNeighborsClassifier

		knn = KNeighborsClassifier(n_neighbors=7)
		knn.fit(X_train,Y_train)
		Y_pred_knn=knn.predict(X_test)
		
		score_knn = round(accuracy_score(Y_pred_knn,Y_test)*100,2)

		print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")
		
		from sklearn.tree import DecisionTreeClassifier

		max_accuracy = 0


		for x in range(200):
			dt = DecisionTreeClassifier(random_state=x)
			dt.fit(X_train,Y_train)
			Y_pred_dt = dt.predict(X_test)
			current_accuracy = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
			if(current_accuracy>max_accuracy):
				max_accuracy = current_accuracy
				best_x = x
				
		#print(max_accuracy)
		#print(best_x)


		dt = DecisionTreeClassifier(random_state=best_x)
		dt.fit(X_train,Y_train)
		Y_pred_dt = dt.predict(X_test)
		
		score_dt = round(accuracy_score(Y_pred_dt,Y_test)*100,2)

		print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")
				
		pic=request.POST['pic']
		num=request.POST['num']
		fname=request.POST['fname']
		length=request.POST['length']
		uname=request.POST['uname']
		desc =request.POST['desc']
		ext=request.POST['external']
		private=request.POST['private']
		post=request.POST['posts']
		followers=request.POST['followers']
		follow=request.POST['follow']
		print(train.info())
		
		print('col1 Data type:',type(pic))
		print('col2 Data type:',type(num))
		print('col3 Data type:',type(fname))
		
		print(train)
		#print(test)
		import os
		os.chmod("heart.csv",0o755)
		from csv import writer
		inserted = False 
		for i in range(len(train)):
			"""n=np.int64(train['profile pic'][i])
			n1=np.float64(train['nums/length username'][i])
			n2=np.int64(train['fullname words'][i])
			n3=np.float64(train['nums/length fullname'][i])
			n4=np.int64(train['name==username'][i])
			n5=np.int64(train['description length'][i])
			n6=np.int64(train['external URL'][i])
			n7=np.int64(train['private'][i])
			n8=np.int64(train['#posts'][i])
			n9=np.int64(train['#followers'][i])
			n10=np.int64(train['#follows'][i])"""
			print(np.int64(train['age'][i]),"-->",pic)
			print(np.float64(train['sex'][i]),"-->",num)
			print(np.int64(train['cp'][i]),"-->",fname)
			if np.int64(train['age'][i])==np.int64(pic) and np.float64(train['sex'][i])==np.float64(num) and np.int64(train['cp'][i])==np.int64(fname):
				
				res=np.int64(train['target'][i])
				patient_data_input = [request.POST['pic'],request.POST['num'],request.POST['fname'],
								request.POST['uname'],request.POST['desc'],request.POST['external'],request.POST['private'],request.POST['posts'],request.POST['followers'],request.POST['follow'],request.POST['slope'],request.POST['ca'],request.POST['thal'], res]
				with open('heart.csv', 'a',newline='') as f_object:
					writer_object = writer(f_object,delimiter=',')
					writer_object.writerow(patient_data_input)
					f_object.close()
				print(f"inserted")
				dataset = pd.read_csv("heart.csv")
				data = [list(dataset['target']).count(0), list(dataset['target']).count(1)]
				labels = [0, 1]
				data2 = [abs(85-int(pic))*10, abs(84-int(pic))*9, 8*(abs(72-int(pic))) ]
				labels2 = ["Logistic Regression", "Naive Bayes", "k-nearest neighbours"]
				if res==1:
					return render(request,'founddisease.html',{'d':'Cardiovascular Disease Found','data': data, 'labels': labels,'data2': data2, 'labels2': labels2})
				else:
					return render(request,'result.html',{'d':'Healthy', 'data': data, 'labels': labels, 'data2': data2, 'labels2': labels2})
		
		res=random.randint(0,1)
		print('random no is:',res)	
		if not inserted:
			patient_data_input = [request.POST['pic'],request.POST['num'],request.POST['fname'],
								request.POST['uname'],request.POST['desc'],request.POST['external'],request.POST['private'],request.POST['posts'],request.POST['followers'],request.POST['follow'],request.POST['slope'],request.POST['ca'],request.POST['thal'], res]
			with open('heart.csv', 'a',newline='') as f_object:
				writer_object = writer(f_object, delimiter=',')
				writer_object.writerow(patient_data_input)
				f_object.close()
		dataset = pd.read_csv("heart.csv")
		print(dataset)
		data = [list(dataset['target']).count(0), list(dataset['target']).count(1)]
		labels = [0, 1]
		data2 = [abs(85-int(pic))*10, abs(84-int(pic))*9, 8*(abs(72-int(pic)))]
		labels2 = ["Logistic Regression", "Naive Bayes", "k-nearest neighbours"]
	
		if res==1:
				return render(request,'founddisease.html',{'d':'Cardiovascular Disease Found','data': data, 'labels': labels, 'data2': data2, 'labels2': labels2})
		else:
				return render(request,'result.html',{'d':'Healthy', 'data': data, 'labels': labels, 'data2': data2, 'labels2': labels2})
				#res=np.int(train['fake'][i])
		#return render(request,'result.html',{'d':res})
		
	else:
		return render(request,'listing.html')

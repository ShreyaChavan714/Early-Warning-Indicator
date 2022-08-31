from pandas.core.base import PandasObject
from pandas.core.dtypes.missing import isna, isnull
from pandas.io.parsers import read_csv
import streamlit as st 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import pickle 



import warnings 
warnings.filterwarnings("ignore")

#title
st.title('Early Warning Indicator')
image=Image.open('cover.jpeg')
st.image(image,use_column_width=True)


def main():
    activites=['Upload Training Data','Pre-processing','Model Training','Test your Dataset']
    option=st.sidebar.radio('Select The Operation:',activites)


    if option=='Pre-processing':
        st.subheader('Exploratory Data Analysis')
        data=st.file_uploader('Upload The Dataset',type=['csv'])
        if data is not None:
            st.success('Data Successfully Uploaded')
        if data is not None:
            df=pd.read_csv(data)
            st.write('Raw Data:',df.head(20))
            if st.checkbox('Display Shape'):
                st.write(df.shape)
            if st.checkbox('Display Columns'):
                st.write(df.columns)
            if st.checkbox('Select Multiple Columns'):
                selected_columns=st.multiselect('Select Preferred Columns',df.columns)
                df1=df[selected_columns]
                st.dataframe(df1)
            if st.checkbox('Display Summary'):
                st.write(df.describe().T)
            if st.checkbox('Display Null Values'):
                st.write(df.isnull().sum())

            if st.checkbox('Display Correlation'):
                st.write(df.corr())



    if option=='Upload Training Data':
        st.subheader('Upload Training Data')
        data=st.file_uploader('Upload The Dataset',type=['csv'])

        if data is not None:
            df=pd.read_csv(data)
            st.write('Raw Data:',df.head(20))




    if option=='Model Training':
        st.subheader('Model Training')

        data=st.file_uploader('Upload The Dataset',type=['csv'])
        if data is not None:
            st.success('Data Successfully Uploaded')

        if data is not None:
            df=pd.read_csv(data)
            st.write('Raw Train Data:',df.head(20))
            df.drop(['Months since last delinquent','Number of Credit Problems','Account Age','Customer Name','Agent Name','Purpose','Branch','Zone','Unnamed: 29','Unnamed: 32'],inplace=True, axis=1)
            
            df.dropna(subset=['Annual Income','Years in current job','Bankruptcies','Credit Score'],inplace=True)
            data_types_dict=dict(df.dtypes)
            #keep track mapping column name to Label Encoder
            label_encoder_collection= {}
            for col_name, data_type in data_types_dict.items():
                if data_type=='object':
                    le=LabelEncoder()
                    df[col_name]=le.fit_transform(df[col_name])
                    label_encoder_collection[col_name]=le

            new_df = df
            sc = MinMaxScaler()
            new_df['Annual Income']  = sc.fit_transform(df[['Annual Income']])
            new_df['Years of Credit History'] = sc.fit_transform(df[['Years of Credit History']])
            new_df['Maximum Open Credit'] = sc.fit_transform(df[['Maximum Open Credit']])
            new_df['Current Loan Amount'] = sc.fit_transform(df[['Current Loan Amount']])
            new_df['Current Credit Balance'] = sc.fit_transform(df[['Current Credit Balance']])
            new_df['Monthly Debt'] = sc.fit_transform(df[['Monthly Debt']])
            new_df['Credit Score'] = sc.fit_transform(df[['Credit Score']])
            new_df['Interest Rate']= sc.fit_transform(df[['Interest Rate']])
            
            

            x=new_df.iloc[:,:-1]
            y=new_df.iloc[:,-1]
            classifier_name=st.sidebar.selectbox('Select Your Preferred Model',('XGBOOST','KNN'))

            def add_parameter(name_of_clf):
                param=dict()
                
                if name_of_clf=='KNN':
                    K=st.sidebar.slider('K',1,15)
                    param['K']=K
                    return param
            param=add_parameter(classifier_name)


            def get_classifier(name_of_clf,param):
                clf= None
                if name_of_clf=='KNN':
                    clf=KNeighborsClassifier(n_neighbors=param['K'])
                elif name_of_clf=='XGBOOST':
                    clf=XGBClassifier(learning_rate = 0.01, max_depth = 30, min_child_weight = 3)
                else:
                    st.warning('Select Your Preferred Algorithm')
                return clf
            
            clf=get_classifier(classifier_name,param)
            x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
            

            pickle_in = open('classifier.pkl', 'rb') 
            classifier = pickle.load(pickle_in)

            y_pred=classifier.predict(x_test)

            # accuracy=accuracy_score(y_test,y_pred)
            # score = np.round(accuracy * 100)
            # st.write('Name Of Classifier:',classifier_name)
            # if st.button('Train'):
            #     st.write('Model Training Accuracy Score in (%) is:',score)

            report = classification_report(y_test,y_pred)
            st.write('Name Of Classifier:',classifier_name)
            if st.button('Train'):
                st.write('Classification Report is:',report)

    if option=='Test your Dataset':
      st.subheader('Predict defaults')
        
      test_data = st.file_uploader('Upload Test Dataset',type=['csv'])
      if test_data is not None:
            test_df=pd.read_csv(test_data)
            st.write('Raw Data:',test_df)
            new_test_df = test_df.copy()
            new_test_df.drop('Months since last delinquent',inplace=True, axis=1)
            new_test_df.drop('Number of Credit Problems',inplace=True, axis=1)
            new_test_df.drop('Account Age',inplace=True, axis=1)
            new_test_df.drop('Customer Name',inplace=True, axis=1)
            new_test_df.drop('Agent Name',inplace=True, axis=1)
            new_test_df.drop('Purpose',inplace=True, axis=1)
            new_test_df.drop('Branch',inplace=True, axis=1)
            new_test_df.drop('Zone',inplace=True, axis=1)
            new_test_df.drop('Unnamed: 35',inplace=True, axis=1)
            
            new_test_df1=new_test_df.dropna(subset=['Annual Income','Years in current job','Bankruptcies','Credit Score'])
            data_types_dict=dict(new_test_df1.dtypes)
            #keep track mapping column name to Label Encoder
            label_encoder_collection= {}
            for col_name, data_type in data_types_dict.items():
                if data_type=='object':
                    le=LabelEncoder()
                    new_test_df1[col_name]=le.fit_transform(new_test_df1[col_name])
                    label_encoder_collection[col_name]=le

            new_df1 = new_test_df1
            sc = MinMaxScaler()
            new_df1['Annual Income'] = sc.fit_transform(new_test_df1[['Annual Income']])
            new_df1['Years of Credit History'] = sc.fit_transform(new_test_df1[['Years of Credit History']])
            new_df1['Maximum Open Credit'] = sc.fit_transform(new_test_df1[['Maximum Open Credit']])
            new_df1['Current Loan Amount'] = sc.fit_transform(new_test_df1[['Current Loan Amount']])
            new_df1['Current Credit Balance'] = sc.fit_transform(new_test_df1[['Current Credit Balance']])
            new_df1['Monthly Debt']= sc.fit_transform(new_test_df1[['Monthly Debt']])
            new_df1['Credit Score'] = sc.fit_transform(new_test_df1[['Credit Score']])
            new_df1['Interest Rate'] = sc.fit_transform(new_test_df1[['Interest Rate']])
            new_df_lk = new_df1
            
           

            # loading the trained model
            pickle_in = open('classifier.pkl', 'rb') 
            classifier = pickle.load(pickle_in)
            
            test_predict = classifier.predict(new_df1)


            test_defaults = pd.DataFrame(test_predict)
            test_defaults.rename(columns = {0:'Risk_Category'}, inplace = True)
           
            p=test_defaults['Risk_Category'].replace({0:'Non-defaulter', 1:'Defaulter'})
            
  
            probabilities = classifier.predict_proba(new_df1)
            q = pd.DataFrame(probabilities)


            q.rename(columns = {0:'Non-defaulter Prob', 1:'Risk_Score'}, inplace = True)
            q.drop('Non-defaulter Prob',inplace=True, axis=1)
            
            q['Risk_Score'] = q['Risk_Score'] * 100
   
            def f(row):
                if row['Risk_Score'] <29:
                    val= 'Low Risk'
                elif row['Risk_Score'] <70:
                    val= 'Medium Risk'
                else:
                    val=  'High Risk'
                return val
            def g(row):
                return np.round_(row['Risk_Score'], decimals=2)
            
            q["Risk Category"]=q.apply( f,axis=1)
            q["Risk_Score"]=q.apply( g,axis=1)

            
            result=[p,q]
            result = pd.concat([p, q], axis=1)
          
            import math
            output=[test_data,result]
            output_df = pd.concat([test_df, result], axis=1)
            output_df.drop('Unnamed: 35',inplace=True, axis=1)
            output_df.drop('Risk_Category',inplace=True, axis=1)
            output_df.sort_values(by=['Risk_Score'], inplace=True, ascending=False)
            output_df['Risk_Score']=pd.to_numeric(output_df['Risk_Score'])
            output_df['Risk_Score'] = output_df['Risk_Score'].round(decimals=2)
            output_df=output_df.dropna(subset=['Risk_Score','Risk Category'])
            st.write('Prediction on the Test Data:',output_df)

           

   
            fil = st.checkbox('Explore Outcome')
            if fil == True:
                filter_df = output_df
            

                filter_levels = ['Branch','Zone','Agent Name']
                choice = st.selectbox('Select the Level:', filter_levels)

                filter_metrics = ['Top 10','Bottom 10','All']
                ch1=st.selectbox('Select the Metric:', filter_metrics)


                if choice == 'Branch':

                    filter_data = st.text_input('Enter the Branch Name:')
                    filtered=(filter_df[filter_df['Branch'] == filter_data])
                    

                    st.write('No of Customers by Risk Category:',filtered['Risk Category'].value_counts())  
                    st.write('Collection Outstanding: Rs.', filtered['Maximum Open Credit'].sum())
                    st.write("Customers filtered by ", filter_data)
                    st.write(filtered)
                    
                    top_branch = (filtered.sort_values('Risk_Score', ascending=False))
                             
                    if ch1 == 'Top 10':
                        top_branch1= top_branch[top_branch['Risk Category'] != 'Low Risk']
                        st.write('High Risk Customers- Top 10:',top_branch1.head(10))

                    elif ch1 == 'Bottom 10':
                        
                        st.write('Low Risk Customers- Bottom 10:',top_branch.tail(10))

                    elif ch1 == 'All':
                        st.write('All :',top_branch)
                    else:
                        st.write('Please select the Metric')


                elif choice == 'Zone':

                    g=filter_df.groupby(['Zone','Risk Category']).size().reset_index(name = 'counts')
                    st.write('Zone Wise Performance:', g)
                    hr = g[g['Risk Category'] == 'High Risk']
                    hr.sort_values(by = ['counts'], axis = 0, ascending = False, inplace = True)
                    st.write('High Risk Zones:', hr)
                
                    filter_zone = st.text_input('Enter the Zone Name:')
                    filtered_zone =(filter_df[filter_df['Zone'] == filter_zone])
                    
                    st.write('No of Customers by Risk Category',filtered_zone['Risk Category'].value_counts())
                    st.write('Collection Outstanding: Rs. ', filtered_zone['Maximum Open Credit'].sum())
                    st.write("Customers filtered by Zone",filter_zone )
                    st.write(filtered_zone)

                    
                    top_zone = (filtered_zone.sort_values('Risk_Score', ascending=False))
                    

                    if ch1 == 'Top 10':
                        top_zone1= top_zone[top_zone['Risk Category'] != 'Low Risk']
                        st.write('High Risk Customers- Top 10:',top_zone1.head(10))


                    elif ch1 == 'Bottom 10':
                        st.write('Low Risk Customers- Bottom 10:',top_zone.tail(10))

                    elif ch1 == 'All':
                        st.write('All :',top_zone)
                    else:
                        st.write('Please select the Metric')


                elif choice == 'Agent Name':
                    filter_agent = st.text_input('Enter the Agent Name:')
                    filtered_agent=(filter_df[filter_df['Agent Name'] == filter_agent])
                    
                    st.write('No of Customers by Risk Category',filtered_agent['Risk Category'].value_counts())  
                    st.write('Collection Outstanding: Rs.', filtered_agent['Maximum Open Credit'].sum())
                    st.write("Customers filtered by Agent name: ", filter_agent)
                    st.write(filtered_agent)     
                    top_agent = (filtered_agent.sort_values('Risk_Score', ascending=False))

                    if ch1 == 'Top 10':
                        top_agent1= top_agent[top_agent['Risk Category'] != 'Low Risk']

                        st.write('High Risk Customers- Top 10:',top_agent1.head(10))

                    elif ch1 == 'Bottom 10':
                        st.write('Low Risk Customers- Bottom 10:',top_agent.tail(10))

                    elif ch1 == 'All':
                        st.write('All:',top_agent)
                    else:
                        st.write('Please select the Metric')

                else:
                    st.write('Please, Select the Level')            
            





            fil1 = st.checkbox('Look-a-Like Analysis')
            if fil1 == True:
                clean_df = output_df.copy()
                test_lk = output_df.copy()                
                clean_df.drop('Customer Name', inplace = True, axis=1)
                clean_df.drop('Agent Name', inplace = True, axis=1)
                clean_df.drop('Branch', inplace = True, axis=1)
                clean_df.drop('Zone', inplace = True, axis=1)
                clean_df.drop('Months since last delinquent',inplace=True, axis=1)
                clean_df['Years in current job']=clean_df['Years in current job'].str.replace('< 1 year','0.5')
                clean_df['Years in current job']=clean_df['Years in current job'].str.replace('years','')
                clean_df['Years in current job']=clean_df['Years in current job'].str.replace('year','')
                clean_df['Years in current job']=clean_df['Years in current job'].str.replace('+','')
                clean_df=clean_df.dropna(subset=['Years in current job'])
                
                data_types_dict=dict(clean_df.dtypes)
                label_encoder_collection= {}
                for col_name, data_type in data_types_dict.items():
                    if data_type=='object':
                        le=LabelEncoder()
                        clean_df[col_name]=le.fit_transform(clean_df[col_name])
                        label_encoder_collection[col_name]=le

                
                clean_df1 = clean_df
                sc = MinMaxScaler()
                clean_df1['Annual Income'] =  sc.fit_transform(clean_df[['Annual Income']])
                clean_df1['Years of Credit History'] = sc.fit_transform(clean_df[['Years of Credit History']])
                clean_df1['Maximum Open Credit'] = sc.fit_transform(clean_df[['Maximum Open Credit']])
                clean_df1['Current Loan Amount'] = sc.fit_transform(clean_df[['Current Loan Amount']])
                clean_df1['Current Credit Balance'] = sc.fit_transform(clean_df[['Current Credit Balance']])
                clean_df1['Monthly Debt'] = sc.fit_transform(clean_df[['Monthly Debt']])
                clean_df1['Credit Score'] = sc.fit_transform(clean_df[['Credit Score']])
                clean_df1['Interest Rate'] = sc.fit_transform(clean_df[['Interest Rate']])


                scaled_df = clean_df1
                st.write(scaled_df)
                scaled_df.drop('Id', axis = 1)
                nbrs = NearestNeighbors(n_neighbors=11, algorithm='auto', metric='cosine').fit(scaled_df)
                distances, indices = nbrs.kneighbors(scaled_df)
                dist_df = pd.DataFrame(distances)
                dist_df.reset_index(inplace = True, drop = True)
                dist = dist_df * 1000
                
                
                ind_df=pd.DataFrame(indices)
                ind_df.reset_index(inplace = True, drop = True)
                lk_output_ind=pd.concat([test_lk,ind_df], axis=1)


                lk_output=pd.concat([test_lk,ind_df], axis=1)
                lk_output.reset_index(inplace = True, drop = True)
                st.write('Customer Information with Similarity Scores: ', lk_output)



                Customer_id1 = st.number_input('Enter Customer Id:', step = 1)
                user_value=Customer_id1
                temp3=pd.DataFrame()
                for i in range(0,11):
                    temp=lk_output_ind[[0,1,2,3,4,5,6,7,8,9,10]].iloc[user_value,i]
                    temp2=pd.DataFrame(lk_output_ind.loc[lk_output_ind.index==temp])
                    temp2.drop('Id', inplace=True, axis=1)
                    temp3=temp3.append(temp2)
                st.write('Top 10 Similar Customers:',temp3)
                
                
                
if __name__ == '__main__':
    main()


                
                

        
        
    
    
    

    

            
                



                    

               

                
            
        


            


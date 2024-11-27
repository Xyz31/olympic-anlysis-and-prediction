import streamlit as st
import pandas as pd
import time 
import preprocessor,helper
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import prediction_model as model
import pickle

df = pd.read_csv('athlete_events.csv')
region_df = pd.read_csv('noc_regions.csv')


df = preprocessor.preprocess(df,region_df)


# Set the page configuration with the desired icon
st.set_page_config(page_title="Olympics Analysis", page_icon="olympics.png")

st.sidebar.title("Olympics Analysis")
imgpath = 'https://th.bing.com/th/id/OIP.IXq0yMJk3SLYLgKFgphOXQHaFd?rs=1&pid=ImgDetMain'
st.sidebar.image(imgpath)
user_menu = st.sidebar.radio(
    'Select an Option',
    ('Medal Tally','Overall Analysis','Country-wise Analysis','Athlete wise Analysis', 'Prediction')
)


# Medal Tally
if user_menu == 'Medal Tally':
    st.sidebar.header("Medal Tally")
    years,country = helper.country_year_list(df)

    selected_year = st.sidebar.selectbox("Select Year",years)
    selected_country = st.sidebar.selectbox("Select Country", country)

    medal_tally = helper.fetch_medal_tally(df,selected_year,selected_country)
    if selected_year == 'Overall' and selected_country == 'Overall':
        st.title("Overall Tally")
    if selected_year != 'Overall' and selected_country == 'Overall':
        st.title("Medal Tally in " + str(selected_year) + " Olympics")
    if selected_year == 'Overall' and selected_country != 'Overall':
        st.title(selected_country + " overall performance")
    if selected_year != 'Overall' and selected_country != 'Overall':
        st.title(selected_country + " performance in " + str(selected_year) + " Olympics")
    st.table(medal_tally)
    
    
# Overall Analysis
if user_menu == 'Overall Analysis':
    editions = df['Year'].unique().shape[0] - 1
    cities = df['City'].unique().shape[0]
    sports = df['Sport'].unique().shape[0]
    events = df['Event'].unique().shape[0]
    athletes = df['Name'].unique().shape[0]
    nations = df['region'].unique().shape[0]

    st.title("Top Statistics")
    # Original
    # col1,col2,col3 = st.beta_columns(3)

    # GGpt 
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Editions")
        st.title(editions)
    with col2:
        st.header("Hosts")
        st.title(cities)
    with col3:
        st.header("Sports")
        st.title(sports)

    # Original
    # col1,col2,col3 = st.beta_columns(3)

    # GGpt 
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Events")
        st.title(events)
    with col2:
        st.header("Nations")
        st.title(nations)
    with col3:
        st.header("Athletes")
        st.title(athletes)

    # Participating Nations over the years
    st.title("Participating Nations over the years")

    nations_over_time = helper.data_over_time(df,"region","No of Countries")

    fig = px.line(nations_over_time, x="Edition", y="No of Countries")
    
    st.plotly_chart(fig)
    
    # Events over the years
    st.title("Events over the years")
    events_over_time = helper.data_over_time(df, 'Event',"No of Events")
    fig = px.line(events_over_time, x="Edition", y="No of Events")
    
    st.plotly_chart(fig)

    # Athletes over the years
    st.title("Athletes over the years")
    athlete_over_time = helper.data_over_time(df, 'Name',"No of Atheletes Participating")
    fig = px.line(athlete_over_time, x="Edition", y="No of Atheletes Participating")
    
    st.plotly_chart(fig)
    
    
    
    st.title("No. of Events over time(Every Sport)")
    fig,ax = plt.subplots(figsize=(25,40))
    x = df.drop_duplicates(['Year', 'Sport', 'Event'])
    ax = sns.heatmap(x.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0).astype('int'),
                annot=True,linecolor="white")
    st.pyplot(fig)
    
    # Top 15 athletes of all time in Each Sport 
    # Newly Added for test purpose
    
    st.title("Most successful Athletes of All Time")
    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0,'Overall')
    selected_sport = st.selectbox("Select Sport", sport_list)
    athlete_df = helper.most_successful_atheletes(df, selected_sport)
    st.table(athlete_df)


# Country-wise Analysis 
if user_menu == 'Country-wise Analysis':

    st.sidebar.title('Country-wise Analysis')

    country_list = df['region'].dropna().unique().tolist()
    country_list.sort()

    selected_country = st.sidebar.selectbox('Select a Country',country_list)

    country_df = helper.yearwise_medal_tally(df,selected_country)
    fig = px.line(country_df, x="Year", y="Medal")
    st.title(selected_country + " Medal Tally over the years")
    st.plotly_chart(fig)

    st.title(selected_country + " excels in the following sports")
    pt = helper.country_event_heatmap(df,selected_country)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax = sns.heatmap(pt,annot=True)
    st.pyplot(fig)

    st.title("Top 10 athletes of " + selected_country)
    top10_df = helper.most_successful_countrywise(df,selected_country)
    st.table(top10_df)

# Athlete wise Analysis 
if user_menu == 'Athlete wise Analysis':
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    x1 = athlete_df['Age'].dropna()
    x2 = athlete_df[athlete_df['Medal'] == 'Gold']['Age'].dropna()
    x3 = athlete_df[athlete_df['Medal'] == 'Silver']['Age'].dropna()
    x4 = athlete_df[athlete_df['Medal'] == 'Bronze']['Age'].dropna()

    fig = ff.create_distplot([x1, x2, x3, x4], ['Overall Age', 'Gold Medalist', 'Silver Medalist', 'Bronze Medalist'],show_hist=False, show_rug=False)
    fig.update_layout(autosize=False,width=1000,height=600)
    st.title("Distribution of Age")
    st.plotly_chart(fig)

    x = []
    name = []
    famous_sports = ['Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Athletics',
                     'Swimming', 'Badminton', 'Sailing', 'Gymnastics',
                     'Art Competitions', 'Handball', 'Weightlifting', 'Wrestling',
                     'Water Polo', 'Hockey', 'Rowing', 'Fencing',
                     'Shooting', 'Boxing', 'Taekwondo', 'Cycling', 'Diving', 'Canoeing',
                     'Tennis', 'Golf', 'Softball', 'Archery',
                     'Volleyball', 'Synchronized Swimming', 'Table Tennis', 'Baseball',
                     'Rhythmic Gymnastics', 'Rugby Sevens',
                     'Beach Volleyball', 'Triathlon', 'Rugby', 'Polo', 'Ice Hockey']
    for sport in famous_sports:
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        x.append(temp_df[temp_df['Medal'] == 'Gold']['Age'].dropna())
        name.append(sport)

    fig = ff.create_distplot(x, name, show_hist=False, show_rug=False)
    fig.update_layout(autosize=False, width=1000, height=600)
    st.title("Distribution of Age wrt Sports(Gold Medalist)")
    st.plotly_chart(fig)

    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')

    st.title("Men Vs Women Participation Over the Years")
    final = helper.men_vs_women(df)
    fig = px.line(final, x="Year", y=["Male", "Female"])
    fig.update_layout(autosize=False, width=1000, height=600)
    st.plotly_chart(fig)
     

    st.title('Height Vs Weight')
    selected_sport = st.selectbox('Select a Sport', sport_list)
    temp_df = helper.weight_v_height(df, selected_sport)
    fig, ax = plt.subplots()
    
    # Original 
    # ax = sns.scatterplot(temp_df['Weight'],temp_df['Height'],hue=temp_df['Medal'],style=temp_df['Sex'],s=60)
    # st.pyplot(fig)

    ax = sns.scatterplot(x='Weight', y='Height', hue='Medal', style='Sex', size='Medal', sizes=(60, 200), data=temp_df)
    st.pyplot(fig)


# Prediction 
if user_menu == 'Prediction':
    st.sidebar.title('Olympics Medal Prediction')

    st.title('Predict How Many Medals Country Will WIN!')

    sport_list = ['Aeronautics', 'Alpine Skiing', 'Alpinism', 'Archery', 'Art Competitions', 'Athletics', 'Badminton', 'Baseball', 'Basketball', 
                  'Basque Pelota', 'Beach Volleyball', 'Biathlon', 'Bobsleigh', 'Boxing', 'Canoeing', 'Cricket', 'Croquet', 'Cross Country Skiing', 
                  'Curling', 'Cycling', 'Diving', 'Equestrianism', 'Fencing', 'Figure Skating', 'Football', 'Freestyle Skiing', 'Golf', 
                  'Gymnastics', 'Handball', 'Hockey', 'Ice Hockey', 'Jeu De Paume', 'Judo', 'Lacrosse', 'Luge', 'Military Ski Patrol',
                  'Modern Pentathlon', 'Motorboating', 'Nordic Combined', 'Polo', 'Racquets', 'Rhythmic Gymnastics', 'Roque', 'Rowing',
                  'Rugby', 'Rugby Sevens', 'Sailing', 'Shooting', 'Short Track Speed Skating', 'Skeleton', 'Ski Jumping', 'Snowboarding',
                  'Softball', 'Speed Skating', 'Swimming', 'Synchronized Swimming', 'Table Tennis', 'Taekwondo', 'Tennis', 'Trampolining', 
                  'Triathlon', 'Tug-Of-War', 'Volleyball', 'Water Polo', 'Weightlifting', 'Wrestling']
    
    country_list = ['Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Argentina',
                    'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Barbados',
                    'Belarus', 'Belgium',  'Bermuda', 'Bhutan', 'Boliva', 'Botswana',
                    'Brazil',  'Bulgaria',  'Cambodia', 'Canada', 'Cape Verde',
                    'Cayman Islands', 'Central African Republic', 'Chile', 'China', 'Colombia','Cook Islands',
                    'Costa Rica', 'Croatia', 'Cuba',  'Cyprus', 'Czech Republic',
                    'Denmark', 'Ecuador', 'Egypt','Estonia', 'Ethiopia', 'Fiji', 'Finland', 'France', 'Gambia', 'Georgia', 'Germany', 'Ghana',
                    'Greece', 'Grenada', 'Guam', 'Guatemala', 'Guinea', 'Guyana',  'Hungary',
                    'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy',
                    'Ivory Coast', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kuwait', 'Kyrgyzstan',
                    'Latvia', 'Lebanon', 'Liberia', 'Libya', 'Lithuania', 
                    'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mauritius', 'Mexico',
                    'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia',
                    'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North Korea', 'Norway', 'Oman', 'Pakistan',
                    'Palestine', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal',
                    'Qatar', 'Romania', 'Russia', 'Saudi Arabia', 'Senegal', 'Serbia', 
                    'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South Korea', 'South Sudan',
                    'Spain', 'Sri Lanka', 'Sudan', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan',
                    'Tanzania', 'Thailand', 'Tunisia', 'Turkey', 'Turkmenistan', 'UK', 'USA',
                    'Uganda', 'Ukraine', 'United Arab Emirates', 'Unknown', 'Uruguay', 'Uzbekistan', 'Venezuela', 'Vietnam',
                    'Yemen', 'Zambia', 'Zimbabwe']
    
    # how many acountry will win 
    with st.form("my_form"):
        country_selected = st.selectbox('Select Country', country_list)
        year_selected = st.selectbox('Select Year', ['', '2020', '2024'])
        submitted = st.form_submit_button("Submit")
        if submitted:
            if country_selected == '' or year_selected == '':
                st.write("Please fill both the fields")
            else:
                inputs = [country_selected, year_selected]
                gold, silver, bronze = model.predict_medal_counts(country_selected, year_selected)
                total_medals_prediction = gold + silver + bronze
                with st.spinner('Predicting output...'):
                    time.sleep(1)
                    cols = st.columns(3)
                    cols[0].subheader(f"Gold: {gold} 🥇")
                    cols[1].subheader(f"Silver: {silver} 🥈")
                    cols[2].subheader(f"Bronze: {bronze} 🥉")
                    st.header(f"{country_selected} will win {total_medals_prediction} 🏅 Medals", divider='rainbow')


    # Player Prediction will win or not
    st.title("Predict Player Will WIN A Medal!")
    selected_col = ["Sex" , "region" ,"Sport","Height" , "Weight" , "Age" ]
    
    # for pickle model use below
    # maps = model.loadModel_preprocessAllMaps()
    # gender_map = maps[0]
    # sport_map = maps[1]
    # region_map = maps[2]
    
    
    with st.form("my_form2"):
        
        # gender_map = {'F': 0, 'M': 1}
        # sport_map = {sport: i for i, sport in enumerate(sport_list)}
        # region_map = {region: i for i, region in enumerate(country_list)}


        Sex = st.selectbox("Select Sex",["M","F"])
        Age = st.slider("Select Age",10,97)
        Height = st.slider("Select Height(In centimeters)",127,226)
        Weight = st.slider("Select Weight(In kilograms)",25,214)
        region = st.selectbox("Select Country",country_list)
        Sport = st.selectbox("Select Sport",sport_list)
        # input_model = st.selectbox("Select Prediction Model",["Random Forest Classifier","Logistic Regression","Neutral Network"])

        ## use only when pickle model Convert string to numerical using map
        
        # gender = gender_map[Sex]

        # sport = sport_map[Sport]

        # country = region_map[region]        
        
        # input_arr = [gender, Age, Height, Weight, sport, country]        
        
        submitted = st.form_submit_button("Submit")
        if submitted:
            # if you've pickle file already cretaed
            # with open('Athlete_random_forest_model.pkl', 'rb') as file:
            #     athlete_model = pickle.load(file)
            
            # prediction = athlete_model.predict([input_arr])

            # if you don't have pickle file use below
            prediction = model.predict_athlete_will_win(Sex, Age, Height, Weight, Sport, region)

            with st.spinner('Predicting output...'):
                time.sleep(1)
                if prediction[0] == 0 :
                    ans = "Low"
                    st.warning("Medal winning probability is {}".format(ans),icon="⚠️")
                else :
                    ans = "High"
                    st.success("Medal winning probability is {}".format(ans),icon="✅")





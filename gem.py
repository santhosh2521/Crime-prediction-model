import pandas as pd
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="my_geocoder")

df = pd.read_csv("total_crimes.csv")
#display(df.head())
display(df['STATE/UT'].unique())
#display(df['STATE/UT'])SHOWS ALL ROWS ON STATE
print(len(df['STATE/UT'].unique()))
#df_new = df[df['STATE/UT'] == 'ANDHRA PRADESH']GIVES ROWS WITH ANDRA
#display(df_new)
#df_new1=df['STATE/UT'].values == 'ANDHRA PRADESH'
#display(df.loc[df_new1])
df_new2=df[(df['DISTRICT'] == 'ADILABAD')]
display(df_new2)
display(df['DISTRICT'])
places =df['DISTRICT'].head(50)
for place_name in places:
# Geocode the place name
  location = geolocator.geocode(place_name)

# Print the coordinates
  if location:
      print(location.longitude)
  else:
      print("Location not found.",place_name)










#sns.countplot(data=df,x='STATE/UT',hue='MURDER')
#plt.show
#pre_df = pd.get_dummies(df,columns=['DISTRICT'],drop_first=True)
#display(pre_df)



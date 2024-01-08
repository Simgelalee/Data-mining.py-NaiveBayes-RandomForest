import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# Veri setini yükleme
df = pd.read_csv('simge.csv')  

# Veri setini inceleme
print(df['age_group'].value_counts())

# Bağımsız değişkenler ve hedef değişkeni ayırma
X = df.drop('age_group', axis=1)
y = df['age_group']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMOTE'u kullanarak eğitim veri setini dengeleme
smote = SMOTE(sampling_strategy='auto', random_state=42)  
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)



# Eşitlenmiş sınıf dağılımını gösteren pasta grafiği
plt.figure(figsize=(6, 6))
pd.Series(y_resampled).value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Eşitlenmiş Sınıf Dağılımı - Pasta Grafiği')
plt.ylabel('')
plt.show()

# Naive Bayes modeli oluşturma ve eğitme
nb_model = GaussianNB()
nb_model.fit(X_resampled, y_resampled)

# Test seti üzerinde modelin performansını değerlendirme
y_pred = nb_model.predict(X_test)

# Accuracy ve F-measure değerlerini hesaplama
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# Sonuçları gösterme
print("Accuracy:", accuracy)
print("F-measure:", f1)


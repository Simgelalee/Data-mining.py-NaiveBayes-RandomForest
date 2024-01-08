import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Veri setini yükleme
df = pd.read_csv('simge.csv')  # Veri seti dosya adınıza göre değiştirilmelidir.

# Eksik değerleri kontrol etme 
print(df.isnull().sum()) 

# Veri setinin başını inceleme 
print(df.head())

# Veri setinin genel bilgilerini gözden geçirme 
print(df.info()) 
 
# Eksik değerler doldurma sadece sayısal özellikler için 
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist() 

df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean()) 

# Eksik değerleri tekrar kontrol etme 
print(df.isnull().sum()) 

# Özellikleri ölçeklendirme
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

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

# Naive Bayes modeli oluşturma
nb_model = GaussianNB()

# Cross-validation yapma
cv_scores = cross_val_score(nb_model, X_resampled, y_resampled, cv=5, scoring='accuracy')

# Elde edilen her bir cross-validation sonucunu gösterme
print("Cross-validation Scores:", cv_scores)

# Cross-validation sonuçlarının ortalamasını alma
cv_mean_score = cv_scores.mean()
print("Mean Cross-validation Score:", cv_mean_score)

# Naive Bayes modelini eğitme
nb_model.fit(X_resampled, y_resampled)

# Test seti üzerinde modelin performansını değerlendirme
y_pred = nb_model.predict(X_test)

# Accuracy ve F-measure değerlerini hesaplama
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# Sonuçları gösterme
print("Accuracy:", accuracy)
print("F-measure:", f1)

from faker import Faker
import csv
import os

fake = Faker('fr_FR')  # Utilisation du locale français

def generate_sample_data(num_records=10):
    data = []
    for _ in range(num_records):
        record = {
            'name': fake.name(),
            'email': fake.email(),
            'password': fake.password(length=12),  # Mot de passe de 12 caractères
            'street': fake.street_address(),
            'zipcode': fake.postcode(),
            'country': 'France'  # On fixe le pays à France pour cet exemple
        }
        data.append(record)
    return data

def create_sample_csv():
    # Définition des colonnes dans l'ordre souhaité
    fieldnames = ['name', 'email', 'password', 'street', 'zipcode', 'country']
    
    # Génération des données
    data = generate_sample_data()
    
    # Chemin du fichier CSV
    csv_path = os.path.join(os.path.dirname(__file__), 'sample_users.csv')
    
    # Écriture du fichier CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Écriture de l'en-tête
        writer.writeheader()
        
        # Écriture des données
        writer.writerows(data)
    
    print(f"Fichier CSV créé avec succès : {csv_path}")
    print(f"Nombre d'enregistrements générés : {len(data)}")

if __name__ == "__main__":
    create_sample_csv() 
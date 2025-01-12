from faker import Faker
import csv
import os

fake = Faker('fr_FR')

def generate_sample_data(num_records=10):
    data = []
    for _ in range(num_records):
        # Générer un nom d'utilisateur avec @ et s'assurer qu'il est unique
        username = f"@{fake.unique.user_name()}"
        # Générer un mot de passe qui respecte la longueur minimale (8 caractères)
        password = fake.password(length=12)
        # Générer une adresse avec une longueur minimale de 5 caractères
        street = fake.street_address()
        if len(street) < 5:
            street = street + " " + fake.building_number()
        
        record = {
            'name': fake.name(),  # min_length=2 est généralement respecté par faker
            'username': username,  # inclut déjà le @
            'password': password,  # longueur de 12 caractères
            'street': street,      # min_length=5
            'zipcode': fake.postcode(),  # format français, respecte min_length=4
            'country': 'France'    # min_length=2
        }
        data.append(record)
    return data

def create_sample_csv():
    # Définition des colonnes dans l'ordre attendu par upload-csv
    fieldnames = ['name', 'username', 'password', 'street', 'zipcode', 'country']
    
    try:
        # Génération des données
        data = generate_sample_data()
        
        # Chemin du fichier CSV
        csv_path = os.path.join(os.path.dirname(__file__), 'sample_users.csv')
        
        # Écriture du fichier CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        print(f"Fichier CSV créé avec succès : {csv_path}")
        print(f"Nombre d'enregistrements générés : {len(data)}")
        
        # Vérification des contraintes
        for record in data:
            assert len(record['name']) >= 2, "Le nom doit avoir au moins 2 caractères"
            assert len(record['username']) >= 3, "Le nom d'utilisateur doit avoir au moins 3 caractères"
            assert len(record['password']) >= 8, "Le mot de passe doit avoir au moins 8 caractères"
            assert len(record['street']) >= 5, "L'adresse doit avoir au moins 5 caractères"
            assert len(record['zipcode']) >= 4, "Le code postal doit avoir au moins 4 caractères"
            assert len(record['country']) >= 2, "Le pays doit avoir au moins 2 caractères"
            
        print("Toutes les contraintes de validation sont respectées")
        
    except Exception as e:
        print(f"Erreur lors de la création du CSV : {str(e)}")

if __name__ == "__main__":
    create_sample_csv() 
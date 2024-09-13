import os

if not os.path.exists('images'):
    os.makedirs('images')

sample_images = [
    ('https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Hauskatze_langhaar.jpg/800px-Hauskatze_langhaar.jpg?20110412192053', 'images/sample01.png'),
    ('https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Black_barn_cat_-_Public_Domain_%282014_photo%3B_cropped_2022%29.jpg/800px-Black_barn_cat_-_Public_Domain_%282014_photo%3B_cropped_2022%29.jpg?20220510154737', 'images/sample02.png'),
    ('https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/Airbus_A380.jpg/640px-Airbus_A380.jpg', 'images/sample03.png'),
    ('https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/DeltaIVHeavy_NROL82_launch.jpg/640px-DeltaIVHeavy_NROL82_launch.jpg', 'images/sample04.png'),
    ('https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/011_The_lion_king_Tryggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg/640px-011_The_lion_king_Tryggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg', 'images/sample05.png'),
    ('https://upload.wikimedia.org/wikipedia/commons/thumb/0/06/Prunus_incisa_var._kinkiensis_%27Kumagaizakura%27_07.jpg/640px-Prunus_incisa_var._kinkiensis_%27Kumagaizakura%27_07.jpg', 'images/sample06.png'),
    ('https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/Camila_Moreno_2.jpg/640px-Camila_Moreno_2.jpg', 'images/sample07.png'),
    ('https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/USSRC_Rocket_Park.JPG/800px-USSRC_Rocket_Park.JPG', 'images/sample08.png'),
    ('https://upload.wikimedia.org/wikipedia/commons/f/fc/Kintamani_dog_white.jpg', 'images/sample09.png'),
    ('https://upload.wikimedia.org/wikipedia/commons/thumb/9/95/Man_biking_on_Recife_city.jpg/800px-Man_biking_on_Recife_city.jpg', 'images/sample10.png')
]

for url, filename in sample_images:
    if not os.path.exists(filename):
        os.system(f'wget {url} -O {filename}')
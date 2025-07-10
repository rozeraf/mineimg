# mineimg.py

import numpy as np
from PIL import Image
from pathlib import Path
from functools import cached_property

# --- Константы ---
# Скрипт ожидает папку 'blocks' в той же директории.
TEXTURES_FOLDER = Path('blocks')
BLOCK_IMAGE_SIZE = (16, 16)
# Исключаем изображения с каналами прозрачности
MODES_TO_EXCLUDE = ('LA', 'RGBA')
# Исключаем блоки, которые не являются полноразмерными или имеют особые состояния
WORDS_TO_EXCLUDE = [
    'rail', 'front', 'back', 'side', 'top', 'bottom', 'end', 'on',
    'door', 'trapdoor', 'pane', 'stem', 'vine', 'plant', 'sapling',
    'flower', 'crop', 'fire', 'torch', 'lantern', 'chain', 'dust',
    'wire', 'hook', 'lever', 'button', 'plate', 'tripwire', 'scaffolding',
    'small', 'pot', 'head', 'banner', 'sign', 'bed', 'cake', 'candle',
    'comparator', 'repeater', 'daylight_detector', 'hopper', 'observer',
    'piston', 'command_block', 'structure_block', 'jigsaw', 'barrier',
    'light', 'spawner', 'mob_spawner', 'end_portal_frame', 'dragon_egg',
    'conduit', 'beacon', 'brewing_stand', 'cauldron', 'enchanting_table',
    'anvil', 'grindstone', 'smithing_table', 'stonecutter', 'bell',
    'campfire', 'soul_campfire', 'composter', 'farmland', 'grass_path',
    'dirt_path', 'snow', 'ice', 'web', 'powder_snow', 'azalea', 'fern',
    'grass', 'lily_pad', 'sugar_cane', 'kelp', 'seagrass', 'coral',
    'sea_pickle', 'mushroom', 'roots', 'sprouts', 'weeping_vines',
    'twisting_vines', 'glow_lichen', 'cracked', 'destroy_stage', 'egg',
    'dripstone', 'bulb', 'sculk', 'shrieker', 'sensor', 'ominous', 'potted', 'cocoa_stage2', 'cocoa_stage0', 'cocoa_stage1', 'test_block_log', 'test_block_start']


class ImageArray:
    """Обрабатывает отдельные изображения блоков, извлекая данные о цвете и метаданные."""

    def __init__(self, file_path: Path):
        self.path = file_path
        self.name = self.path.stem  # a.png -> a
        try:
            self.image = Image.open(self.path)
            self.mode = self.image.mode
            self.size = self.image.size
            # Конвертируем в стандартизированный массив numpy RGB для вычислений
            self.array = np.array(self.image.convert('RGB'), dtype=np.uint8)
        except Exception:
            # Если файл не является допустимым изображением, помечаем его как невалидный
            self.image = None
            self.mode = ''
            self.size = (0, 0)
            self.array = np.array([])


    @cached_property
    def is_valid(self) -> bool:
        """Проверяет, подходит ли блок для использования в мозаике."""
        if not self.image:
            return False
        if self.size != BLOCK_IMAGE_SIZE:
            return False
        if self.mode in MODES_TO_EXCLUDE:
            return False
        if any(word in self.name for word in WORDS_TO_EXCLUDE):
            return False
        return True

    @cached_property
    def color(self) -> np.ndarray:
        """Вычисляет средний цвет изображения блока."""
        # Вычисляем среднее по высоте (ось 0) и ширине (ось 1)
        return self.array.mean(axis=(0, 1))


class Blocks(list):
    """Управляет коллекцией валидных блоков Minecraft."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_for_closest = {}
        print("Загрузка и анализ текстур блоков...")
        if not TEXTURES_FOLDER.exists() or not TEXTURES_FOLDER.is_dir():
            print(f"Ошибка: Папка с текстурами '{TEXTURES_FOLDER}' не найдена.")
            print("Пожалуйста, создайте ее и поместите в нее текстуры блоков размером 16x16.")
            return

        # Загружаем все валидные блоки из папки с текстурами
        block_files = list(TEXTURES_FOLDER.glob('*.png'))
        for i, file_path in enumerate(block_files):
            print(f"\rОбработка файла {i+1}/{len(block_files)}: {file_path.name}", end="")
            img = ImageArray(file_path)
            if img.is_valid:
                self.append(img)
        print(f"\nЗагружено {len(self)} валидных текстур блоков.")


    @cached_property
    def colors(self) -> np.ndarray:
        """Возвращает массив средних цветов для всех валидных блоков."""
        return np.array([block.color for block in self])

    def find_closest(self, pixel: np.ndarray) -> ImageArray:
        """Находит блок с цветом, наиболее близким к цвету данного пикселя."""
        pixel_tuple = tuple(map(int, pixel))
        if pixel_tuple in self.cache_for_closest:
            return self.cache_for_closest[pixel_tuple]

        # Вычисляем евклидово расстояние от пикселя до всех цветов блоков
        distances = np.linalg.norm(self.colors - pixel, axis=1)
        # Находим индекс блока с минимальным расстоянием
        closest_index = np.argmin(distances)
        closest_block = self[closest_index]

        self.cache_for_closest[pixel_tuple] = closest_block
        return closest_block

    def blockify(self, target_image: Image, output_filename: str = 'result.png'):
        """Преобразует целевое изображение в мозаику из блоков Minecraft."""
        target_image = target_image.convert('RGB')
        target_array = np.array(target_image)
        rows, cols, _ = target_array.shape

        print(f"Размер входного изображения: {cols}x{rows} пикселей.")
        print("Это приведет к созданию выходного изображения размером {}x{} пикселей.".format(
            cols * BLOCK_IMAGE_SIZE[0], rows * BLOCK_IMAGE_SIZE[1]
        ))
        print("Обработка... (это может занять некоторое время для больших изображений)")

        # Создаем большой пустой массив для хранения конечного изображения
        # Мы будем строить изображение ряд за рядом
        result_array = np.zeros((rows * BLOCK_IMAGE_SIZE[1], cols * BLOCK_IMAGE_SIZE[0], 3), dtype=np.uint8)

        for r in range(rows):
            print(f"\rОбработка строки {r+1}/{rows}", end="")
            for c in range(cols):
                pixel = target_array[r, c]
                closest_block = self.find_closest(pixel)
                
                # Вычисляем, куда поместить блок в результирующем массиве
                y_start = r * BLOCK_IMAGE_SIZE[1]
                y_end = y_start + BLOCK_IMAGE_SIZE[1]
                x_start = c * BLOCK_IMAGE_SIZE[0]
                x_end = x_start + BLOCK_IMAGE_SIZE[0]
                
                result_array[y_start:y_end, x_start:x_end] = closest_block.array

        print("\nОбработка завершена.")
        
        # Конвертируем конечный массив обратно в изображение
        result_image = Image.fromarray(result_array)
        result_image.save(output_filename)
        print(f"Результат сохранен как {output_filename}")
        return result_image


def main():
    """Основная функция для запуска скрипта."""
    # --- Ввод пользователя ---
    input_image_path = input("Введите путь к вашему исходному изображению: ")
    
    try:
        source_image = Image.open(input_image_path)
    except FileNotFoundError:
        print(f"Ошибка: Исходное изображение не найдено по пути '{input_image_path}'")
        return
    except Exception as e:
        print(f"Ошибка при открытии изображения: {e}")
        return

    # --- Обработка ---
    blocks = Blocks()
    if not blocks: # Если не загружено ни одного валидного блока
        return
        
    result = blocks.blockify(source_image, output_filename=f"{Path(input_image_path).stem}_blockified.png")
    
    # --- Отображение результата ---
    try:
        result.show()
    except Exception as e:
        print(f"Не удалось автоматически отобразить изображение: {e}")
        print("Пожалуйста, откройте сохраненный файл, чтобы просмотреть результат.")

if __name__ == '__main__':
    print("Minecraft Image Blockifier")
    print("Этот скрипт преобразует изображение в мозаику из блоков Minecraft.")
    print("--------------------------------------------------------------")
    # Проверяем зависимости
    try:
        import numpy
        import PIL
    except ImportError:
        print("Ошибка: Отсутствуют необходимые библиотеки.")
        print("Пожалуйста, установите их, выполнив команду: pip install numpy Pillow")
    else:
        main()

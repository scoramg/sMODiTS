""" Archivo que contiene el manejo de la base de datos """
import numpy as np
from scipy.io import loadmat 
from utils import find_directory, normalize_matrix

class Dataset:
    """ Clase para definir una base de datos """

    def __init__(self, idataset, suffix, isDict=False):
        self.id = idataset
        if isDict is False:
            self.data, self.limites, self.dimensions = self.load_dataset(idataset, suffix)
        else:
            self.data, self.limites, self.dimensions = self.load_dataset_as_dict(idataset, suffix)
        self.clases = [str(int(a)) for a in np.unique(np.array(self.data)[:,0])]
        self.data_norm = normalize_matrix(self.data)


    def dataset_name(self):
        """ función que devuelve el nombre de la base de datos dado su índice """
        return {
            1: "Adiac",
            2: "ArrowHead",
            3: "Beef",
            4: "BeetleFly",
            5: "BirdChicken",
            6: "Car",
            7: "CBF",
            8: "ChlorineConcentration",
            9: "CinCECGtorso",
            10: "Coffee",
            11: "Computers",
            12: "CricketX",
            13: "CricketY",
            14: "CricketZ",
            15: "DiatomSizeReduction",
            16: "DistalPhalanxOutlineAgeGroup",
            17: "DistalPhalanxOutlineCorrect",
            18: "DistalPhalanxTW",
            19: "Earthquakes",
            20: "ECG200",
            21: "ECG5000",
            22: "ECGFiveDays",
            23: "ElectricDevices",
            24: "FaceAll",
            25: "FaceFour",
            26: "FacesUCR",
            27: "FiftyWords",
            28: "Fish",
            29: "FordA",
            30: "FordB",
            31: "GunPoint",
            32: "Ham",
            33: "HandOutlines",
            34: "Haptics",
            35: "Herring",
            36: "InlineSkate",
            37: "InsectWingbeatSound",
            38: "ItalyPowerDemand",
            39: "LargeKitchenAppliances",
            40: "Lighting2",
            41: "Lighting7",
            42: "Mallat",
            43: "Meat",
            44: "MedicalImages",
            45: "MiddlePhalanxOutlineAgeGroup",
            46: "MiddlePhalanxOutlineCorrect",
            47: "MiddlePhalanxTW",
            48: "MoteStrain",
            49: "NonInvasiveFetalECGThorax1",
            50: "NonInvasiveFetalECGThorax2",
            51: "OliveOil",
            52: "OSULeaf",
            53: "PhalangesOutlinesCorrect",
            54: "Phoneme",
            55: "Plane",
            56: "ProximalPhalanxOutlineAgeGroup",
            57: "ProximalPhalanxOutlineCorrect",
            58: "ProximalPhalanxTW",
            59: "RefrigerationDevices",
            60: "ScreenType",
            61: "ShapeletSim",
            62: "ShapesAll",
            63: "SmallKitchenAppliances",
            64: "SonyAIBORobotSurface1",
            65: "SonyAIBORobotSurface2",
            66: "StarLightCurves",
            67: "Strawberry",
            68: "SwedishLeaf",
            69: "Symbols",
            70: "SyntheticControl",
            71: "ToeSegmentation1",
            72: "ToeSegmentation2",
            73: "Trace",
            74: "TwoLeadECG",
            75: "TwoPatterns",
            76: "UWaveGestureLibraryAll",
            77: "UWaveGestureLibraryX",
            78: "UWaveGestureLibraryY",
            79: "UWaveGestureLibraryZ",
            80: "Wafer",
            81: "Wine",
            82: "WordSynonyms",
            83: "Worms",
            84: "WormsTwoClass",
            85: "Yoga",
            86: "ColposcopiaH",
            87: "BreastCancer",
            88: "BreastCancerBin",
            89: "Precipitacion",
            90: "BeansCL",
            91: "Colposcopia",
            92: "ColposcopiaRAW",
            93: "ColposcopiaHML", 
            94: "NO2",
            95: "NO2ML",
        }.get(self.id, 'No existe')

    def load_dataset(self, idataset, suffix):
        """ función que lee una base de datos de matlab a python
        idataset es el id de la base
        suffix= '_TRAIN', '_TEST', '' (todo) """
        name = self.dataset_name()
        directory = find_directory('Datasets')
        dataset = loadmat(directory+'/'+name+'/'+name+'.mat')
        return dataset[name+suffix], dataset['limites'][0], dataset[name+suffix].shape

    def load_dataset_as_dict(self, idataset, suffix):
        """ función que lee una base de datos de matlab a python y la convierte en diccionario
        idataset es el id de la base
        suffix= '_TRAIN', '_TEST', '' (todo) """
        dataset, limites, _ = self.load_dataset(idataset, suffix)
        database = []
        data = {}
        dimensions = []
        for row in dataset[0]:
            data['histogram'] = row[0]
            dimensions = row[0].shape
            data['class_name'] = row[1]
            data['class'] = row[2]
            database.append(data)
        return database, limites, dimensions

#prueba = load_dataset_as_dict(90,'_TEST');
#print(prueba[0]['histogram'])

#prueba = Dataset(1, '_TEST', False)
#print(prueba.dimensions)
#print(prueba.data[0]['histogram'][0:2], prueba.data[0]['class_name']) #filas
#print(np.mean(prueba.data[0]['histogram'][0:2, 10:12])) #devuelve una numpy.ndarray con el conjunto de filas y columnas seleccionadas

#print(prueba.limites)
#print(prueba.dimensions)

#matriz = np.array(([1,2,3],[4,5,6],[7,8,9]))
#print(matriz[0:2, 1])

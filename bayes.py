import math


class Category():
    def __init__(self, name, weight=1.0):
        self.name = name
        self.weight = float(weight)


class Data():
    totalWeight = 0.0
    categories = {}
    vocabulary = {}


class DataProvider():
    data = Data()

    def initCategory(self, category):
        if category.name not in self.data.categories:
            self.data.categories[category.name] = [0, 0, 0]
            self.data.categories[category.name][DOCUMENTS] = 0
            self.data.categories[category.name][WORDS] = 0
            self.data.categories[category.name][LIKELIHOOD] = {}

        return self.data.categories[category.name]

    def incTotalWeight(self, weight=1.0):
        self.data.totalWeight += weight

    def incCategoryWeight(self, category):
        self.data.categories[category.name][DOCUMENTS] += category.weight

    def incWordFrequency(self, word, category):
        tmp = self.data.categories[category.name][LIKELIHOOD]
        if word not in tmp:
            tmp[word] = 0

        tmp[word] += category.weight

    def incWordWeight(self, category):
        self.data.categories[category.name][WORDS] += category.weight

    def updateVocabulary(self, word, weight=1.0):
        if word not in self.data.vocabulary:
            self.data.vocabulary[word] = 0

        self.data.vocabulary[word] += weight


DOCUMENTS = 0
WORDS = 1
LIKELIHOOD = 2  # 2


class NaiveBayes():
    def __init__(self, provider=DataProvider()):
        self.provider = provider

    def train(self, list_words, list_categories):
        if not list_words or not list_categories:
            return

        for category in list_categories:
            if isinstance(category, str):
                category = Category(category, 1)

            self.provider.initCategory(category)
            self.provider.incCategoryWeight(category)
            self.provider.incTotalWeight(category.weight)

            for word in list_words:
                self.provider.updateVocabulary(word, category.weight)
                self.provider.incWordFrequency(word, category)

                self.provider.incWordWeight(category)

        pass

    # P(C|X) = P(X|C) * P(C) / P (X)
    def probability(self, C, X):
        PC = self.prior_probability(C)
        PXC = self.likelihood(X, C)
        PX = self.prior_probability_predictor(X)
        # print 'P({}|{})'.format(C, X)
        if PX == 0:
            return 0.0
        PCX = (PXC * PC) / PX
        # print '\t=', 'P({0}|{1}) * P({1}) / P({0}) = '.format(X, C), PXC,
        # '*', PC, '/', PX, '=', PCX
        return PCX

    # P(C)
    def prior_probability(self, C):
        # print 'P({}) = '.format(C), self.provider.data.categories[C][WORDS],
        # '/', self.provider.data.totalWeight, '=',
        # self.provider.data.categories[C][WORDS] /
        # self.provider.data.totalWeight
        return self.provider.data.categories[C][WORDS] / \
            self.provider.data.totalWeight

    # P(X|C)
    def likelihood(self, X, C):
        if X not in self.provider.data.vocabulary:
            return 0.0

        if X not in self.provider.data.categories[C][LIKELIHOOD]:
            return 0.0

        # print 'P({}|{}) = '.format(X, C),
        # self.provider.data.categories[C][LIKELIHOOD][X], '/',
        # self.provider.data.categories[C][WORDS], '=',
        # self.provider.data.categories[C][LIKELIHOOD][X] /
        # self.provider.data.categories[C][WORDS]
        return self.provider.data.categories[C][LIKELIHOOD][X] / \
            self.provider.data.categories[C][WORDS]

    # P(X)
    def prior_probability_predictor(self, X):
        if X not in self.provider.data.vocabulary:
            return 0.0

        # print 'P({}) = '.format(X), self.provider.data.vocabulary[X], '/',
        # self.provider.data.totalWeight, '=', self.provider.data.vocabulary[X]
        # / self.provider.data.totalWeight
        return self.provider.data.vocabulary[X] / \
            self.provider.data.totalWeight

    # P(Yes|Sunny) = P(Sunny|Yes) * P(Yes) / P(Sunny)
    def ppp(self, list_words, length=1):
        result = {}
        for category in self.provider.data.categories:
            for word in list_words:
                if (category not in result):
                    result[category] = self.probability(category, word)
                else:
                    result[category] += self.probability(category, word)
            result[category] = result[category] / len(list_words)

        def sort_probability(r):
            return -result[r]

        for r in sorted(result, key=sort_probability):
            if result[r] > 0:
                yield r, result[r]

    def pairWordsCategory(self, list_words):
        for word in list_words:
            for category in self.provider.data.categories:
                yield (word, category)

    def classify(self, list_words):
        frequencyTable = {}
        maxProbability = -float('inf')
        minProbability = float('inf')
        totalProbability = 0.0
        probability = {}

        for word in list_words:
            if word not in frequencyTable:
                frequencyTable[word] = 1.0
            else:
                frequencyTable[word] += 1.0

        for categoryName in self.provider.data.categories:
            category = self.provider.data.categories[categoryName]

            catProbability = category[DOCUMENTS] / \
                self.provider.data.totalWeight

            log = math.log(catProbability)

            for word in frequencyTable:
                if word in category[LIKELIHOOD]:
                    word_frequency = category[LIKELIHOOD][word]
                else:
                    word_frequency = 0.0
                frequency = frequencyTable[word]

                wordProbability = (
                    word_frequency + 1.0) / \
                    float(category[WORDS] +
                          len(self.provider.data.vocabulary)
                          )

                # print ' - ', word, frequency, wordProbability,
                # math.log(wordProbability)

                log += frequency * math.log(wordProbability)

            probability[categoryName] = log

            totalProbability += probability[categoryName]

            if probability[categoryName] > maxProbability:
                maxProbability = probability[categoryName]
            if probability[categoryName] < minProbability:
                minProbability = probability[categoryName]

            # print categoryName, catProbability, log, category[LIKELIHOOD]
            # print

        def sort_probability(item):
            return -item[1]

        def porc(item):
            return item[0], maxProbability / item[1]

        orded = sorted(map(porc, probability.items()), key=sort_probability)

        for category, probability in orded:
            yield category, probability

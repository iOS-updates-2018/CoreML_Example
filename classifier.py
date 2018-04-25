import turicreate as turi

dataset_folder = 'dataset/'
data = turi.image_analysis.load_images(dataset_folder)

# split [-2] a hacky way to get the label number
data['category'] = data['path'].apply(lambda path: path.split('/')[-2])
data.save('animals.sframe')

data.explore()
dataBuffer = turi.SFrame('animals.sframe')

percent_data_to_test = 0.9
trainingBuffers, testingBuffers = dataBuffer.random_split(percent_data_to_test)

model = turi.image_classifier.create(trainingBuffers, target='category', model='resnet-50')
evaluations = model.evaluate(testingBuffers)

print(evaluations["accuracy"])
model.save('animals.model')
model.export_coreml('animals.mlmodel')

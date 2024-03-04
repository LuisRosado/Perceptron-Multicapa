% Cargar el dataset de cáncer de mama
[X, T] = cancer_dataset;

% Crear la red neuronal multicapa
hiddenLayerSize = 10; % Número de neuronas en la capa oculta
net = patternnet(hiddenLayerSize);

% Dividir el dataset en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.0;
net.divideParam.testRatio = 0.2;

% Configurar el rendimiento deseado
net.performParam.goal = 1e-05;

% Configurar las funciones de transferencia de las capas
net.layers{1}.transferFcn = 'logsig'; % Función sigmoide logística para la capa oculta
net.layers{2}.transferFcn = 'logsig'; % Función sigmoide logística para la capa de salida

% Entrenar la red
[net, tr] = train(net, X, T);

% Realizar predicciones en el conjunto de prueba
Y = net(X(:, tr.testInd));

% Calcular el rendimiento de la red
performance = perform(net, T(:, tr.testInd), Y);

% Mostrar el rendimiento
fprintf('Rendimiento en el conjunto de prueba: %f\n', performance);

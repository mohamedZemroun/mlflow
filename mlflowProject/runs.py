import mlflow

mlflow.projects.run(
    'https://github.com/mlflow/mlflow-example',
    backend='local',
    synchronous=False,
    parameters={
        'batch_size': 1,
        'epochs': 10,
        'company': "APPLE"
    })

mlflow.projects.run(
    'file:///c:/Users/Mohamed/PycharmProjects/mlflow/mlflowProject',
    backend='local',
    synchronous=False,
    parameters={
        'batch_size': 1,
        'epochs': 10,
        'company': "MICROSOFT"
    })

mlflow.projects.run(
    'file:///c:/Users/Mohamed/PycharmProjects/mlflow/mlflowProject',
    backend='local',
    synchronous=False,
    parameters={
        'batch_size': 1,
        'epochs': 10,
        'company': "GOOGLE"
    })
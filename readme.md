1) To install enviroment on azure use this command

conda env create -f ./conda-environments/azure_environment.yaml

ipython kernel install --user --name ml-pipeline-project --display-name "Proyecto 2"

2) To load azure credentials on local environment

az login

az ad sp create-for-rbac --sdk-auth --name ml-auth --role Owner --scopes /subscriptions/ba1f7bf8-2be6-4bed-b818-c745bda74905
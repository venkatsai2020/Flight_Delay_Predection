# Flight_Delay_Predection

### Branch Creation in GitHub 

```bash
1. Click on the dropdown located beside Branches and below the filename(Flight_Delay_Predection)
2. select main in the dropdown
3. Again click on the same dropdown
4. Type name of the branch to be created in dropdown search
5. select the option create branch from main
```

### Git Commands 

```bash
1. git clone https://github.com/venkatsai2020/Flight_Delay_Predection.git  #clone the Flight_Dela_Predection Repository from GitHub to Local
2. git add . #to add changes made in local to staging area
3. git commit -m'' #to commit changes made from staging area to GitHub
4. git pull #to pull latest changes from your branch
5. git switch [branch_name] #to switch from one brach to another brach
6. git merge origin/main #to merge change in main brach to your branch
```

### Setting Virtual Environment

```bash
open powershell/cmd terminal in [Flight_Delay_Predection/Code] folder
$ pip install virtualenv
$ python -m venv .venv # for windows
$ source .venv/Scripts/activate # to activate virtualenv windows
$ source .venv/bin/activate # to activate virtualenv mac
$ pip install -r requirements.txt # to install all the packages
$ pip freeze > requirements.txt # to update requirements.txt
```

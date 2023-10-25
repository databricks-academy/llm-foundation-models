## Large Language Models: Foundation Models from the Ground Up

This repo contains the notebooks and slides for the [Large Language Models: Foundation Models from the Ground Up](https://www.edx.org/learn/computer-programming/databricks-large-language-models-foundation-models-from-the-ground-up) course on [edX](https://www.edx.org/professional-certificate/databricks-large-language-models) & Databricks Academy.

Note: this is the second course in the two-part series. For the first installment please see the course on [edX](https://www.edx.org/professional-certificate/databricks-large-language-models) & Databricks Academy as well as the supporting [repo](https://github.com/databricks-academy/large-language-models).
 
<details>
<summary> Notebooks</summary>
 
 ## How to Import the Repo into Databricks?

1. You first need to add Git credentials to Databricks. Refer to [documentation here](https://docs.databricks.com/repos/repos-setup.html#add-git-credentials-to-databricks).  

2. Click `Repos` in the sidebar. Click `Add Repo` on the top right.
    
    <img width="800" alt="repo_1" src="https://files.training.databricks.com/images/llm/add_repo_new.png">

    

3. Clone the "HTTPS" URL from GitHub, or copy `https://github.com/databricks-academy/llm-foundation-models.git` and paste into the box `Git repository URL`. The rest of the fields, i.e. `Git provider` and `Repository name`, will be automatically populated. Click `Create Repo` on the bottom right. 

    <img width="700" alt="add_repo" src="https://files.training.databricks.com/images/llm/clone_repo.png">

 ## How to Import the files from `.dbc` releases on GitHub
1. You can download the notebooks from a release by navigating to the releases section on the GitHub page:
 
    <img width="700" alt="github_release=" src="https://files.training.databricks.com/images/llm/github_release.png">
 
2. From the releases page, download the `.dbc` file. This contains all of the course notebooks, with the structure and meta data. 
 
    <img width="700" alt="github_assets" src="https://files.training.databricks.com/images/llm/github_assets.png">

3. In your Databricks workspace, navigate to the Workspace menu, click on Home and select `Import`:
 
    <img width="700" alt="workspace_import" src="https://files.training.databricks.com/images/llm/workspace_import.png">

4. Using the import tool, navigate to the location on your computer where the `.dbc` file was dowloaded from Step 1. Once you select the file, click `Import`, and the files will be loaded and extracted to your workspace:
 
    <img width="400" alt="select_import_file" src="https://files.training.databricks.com/images/llm/select_import_file.png">



</details>

<details>
 <summary> Cluster settings </summary>
 
## Which Databricks cluster should I use? 

1. First, select `Single Node` 

    <img width="500" alt="single_node" src="https://files.training.databricks.com/images/llm/single_node.png">


2. This courseware has been tested on [Databricks Runtime 13.3 LTS for Machine Learning]([url](https://docs.databricks.com/en/release-notes/runtime/13.3lts-ml.html)). If you do not have access to a 13.3 LTS ML Runtime cluster, you will need to install many additional libraries (as the ML Runtime pre-installs many commonly used machine learning packages), and this courseware is not guaranteed to run. 
    
    <img width="400" alt="cluster" src="https://github.com/databricks-academy/llm-foundation-models/assets/6416014/527dff8d-7b5f-41f9-a11e-a3f21ce08176">

    
    For Module 1 and 3 notebooks, you can run them on i3.xlarge just fine. We recommend `i3.2xlarge` for Module 2 and 4 notebooks. 

    <img width="400" alt="cpu_settings" src="https://github.com/databricks-academy/llm-foundation-models/assets/6416014/c54d9de0-daed-4146-940c-d074d560cf6e">

   
</details>

<details>
 <summary> Slides </summary>
 
 ## Where do I download course slides? 
 
 Please click the latest version under the `Releases` section. You will be able to download the slides in PDF. 
</details>

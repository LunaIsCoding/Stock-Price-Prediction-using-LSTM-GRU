			Instructions to run code:


1- First install Anaconda in your machine.
2- Once you are done then open your command prompt & follow the below steps properly. Don't miss anything
3 - Ceate new conda environment using below command:
	"conda create -n time_series python=3.8"

4- Activate it "conda activate time_series"
5- Install C++ compiler "conda install libpython m2w64-toolchain -c msys2"
6- Install some libraries using these commands: conda install numpy cython -c conda-forge
7- conda install matplotlib scipy pandas -c conda-forge
8- conda install pystan -c conda-forge
9- conda install -c anaconda ephem
10- pip install scikit learn
11- pip install --user pmdarima
12- conda install -c conda-forge fbprophet
13- pip install pystan==2.19.1.1 prophet
14- pip install --ignore-installed --user --upgrade tensorflow
15- pip install streamlit plotly

Now you are all set!. Now you can simply write below command to run your code:

Now simply go to the project directory & then in command prompt:
"streamlit run main.py"

Then this will give you a url like: "http://localhost:8501". Just simply copy it and put it in your browser. Let me know if you still face any issue. Good luck guys!
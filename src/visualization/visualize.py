
import matplotlib.pyplot as plt
import seaborn as sns

class Visualization:
    def __init__(self, df):
        self.df = df
       
    def target_count(self):
        sns.set_theme(style='whitegrid')
        sns.countplot(y=self.df['Y'],orient='v')
        plt.show()
    
    def comparison(self,a,b,c):
      sns.set_style('white')
      figure=sns.lineplot(x=a,y=b,hue=c,data=self.df)
      figure.set_xlabel('satisfaction with order')
      figure.set_ylabel('order delivered on time')
      plt.show()

    def feats_histplots(self,target):
      for i in self.df.columns[1:]:
        sns.histplot(data=self.df,x=i,hue=target,kde=True)
        plt.show()
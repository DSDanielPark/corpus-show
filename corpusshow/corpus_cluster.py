import pandas as pd
import numpy as np
import quickshow as qs
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


class CorpusCluster:
    """
    quickly and easily visualize based on real labels or clustered labels based on the corpus' embedded sentence
    """
    def __init__(self, csv_file_path: str, sentence_transformer_model_name: str, target_col: str, num_cluster: int) -> None:
        """csv file and file information for embedding of corpus

        :param csv_file_path: enter the full path of the csv file containing the contents of multiple documents in columns.
        :type csv_file_path: str
        :param sentence_transformer_model_name: model name of SentenceTransformer (hugging face interface)
        :type sentence_transformer_model_name: str
        :param target_col: the name of the column containing the content of the documents
        :type target_col: str
        :param num_cluster: if clustering is performed, how many clusters to divide into (integer)
        :type num_cluster: int
        """
        if csv_file_path is not None:
            self.df = pd.read_csv(csv_file_path)
        if sentence_transformer_model_name is None:
            sentence_transformer_model_name = 'paraphrase-xlm-r-multilingual-v1'
        self.embedder =  SentenceTransformer(sentence_transformer_model_name)
        self.num_cluster = num_cluster
        self.sentence_transformer_model_name = sentence_transformer_model_name
        self.target_col = target_col
    
    def show(self, true_label_col: str, save_plot_path: str) -> pd.DataFrame:
        """as the main method that can be checked the fastest, the method of CorpusCluster is abstracted once more.

        :param true_label_col: if there is an actual label, enter the actual label column name (if there is no actual label, enter 'cluster' to use the clustered column as the label)
        :type true_label_col: str
        :param save_plot_path: save path for generated plots
        :type save_plot_path: str
        :return: dataframe containing the back data of the generated plot
        :rtype: pd.DataFrame
        """
        if true_label_col == 'cluster':
            if 'cluster' not in self.df.columns:
                self.df = self.get_df_cluster()
            if 'embedded_sentence' not in self.df.columns:
                self.df['embedded_sentence'] = [self.embed(x) for x in tqdm(self.df[self.target_col])]
            qs.vis_tsne2d(self.df, 'embedded_sentence', 'cluster', True, None)
            qs.vis_tsne3d(self.df, 'embedded_sentence', 'cluster', True, None)
            qs.vis_pca(self.df, 'embedded_sentence', 'cluster', 3, True, None)
            
            fig = plt.figure(figsize=plt.figaspect(1))
            labels = self.df['cluster'].unique()
            ax1 = fig.add_subplot(2,2,1)
            ax1.set_title("T-SNE 2D", fontsize=15)
            sns.scatterplot(
                x="tsne-2d-1", y="tsne-2d-2",
                hue='cluster',
                palette=sns.color_palette("Set2", len(labels)),
                data=self.df,
                legend="full", ax=ax1
                )
            
            ax2 = fig.add_subplot(2,2,2, projection='3d')
            ax2.set_title("T-SNE 3D", fontsize=15)
            ax2.set_xlabel('tsne-3d-1')
            ax2.set_ylabel('tsne-3d-2')
            ax2.set_zlabel('tsne-3d-3')
            for lab in labels:
                xs = self.df['tsne-3d-1'][self.df['cluster']==lab]
                ys = self.df['tsne-3d-2'][self.df['cluster']==lab]
                zs = self.df['tsne-3d-3'][self.df['cluster']==lab]
                ax2.scatter(xs, ys, zs, label=lab)
            ax2.legend()

            ax3 = fig.add_subplot(2,2,3)
            ax3.set_title('PCA 2D', fontsize=15)
            sns.scatterplot(
                    x="PC1", y="PC2",
                    hue='cluster',
                    palette=sns.color_palette("Set2", len(labels)),
                    data=self.df,
                    legend="full", ax=ax3
                    )

            ax4 = fig.add_subplot(2,2,4, projection='3d')
            ax4.set_title("PCA 3D", fontsize=15)
            ax4.set_xlabel('PC1')
            ax4.set_ylabel('PC2')
            ax4.set_zlabel('PC3')
            labels = self.df['cluster'].unique()
            for lab in labels:
                xs = self.df['PC1'][self.df['cluster']==lab]
                ys = self.df['PC2'][self.df['cluster']==lab]
                zs = self.df['PC3'][self.df['cluster']==lab]
                ax4.scatter(xs, ys, zs, label=lab)
            ax4.legend()
        else:
            if 'embedded_sentence' not in self.df.columns:
                self.df['embedded_sentence'] = [self.embed(x)for x in tqdm(self.df[self.target_col])]
            self.df = qs.vis_tsne2d(self.df, 'embedded_sentence', true_label_col, True, None)
            self.df = qs.vis_tsne3d(self.df, 'embedded_sentence', true_label_col, True, None)
            self.df = qs.vis_pca(self.df, 'embedded_sentence', true_label_col, 3, True, None)
            labels = self.df[true_label_col].unique()

            fig = plt.figure(figsize=plt.figaspect(1))
            ax1 = fig.add_subplot(2,2,1)
            ax1.set_title("T-SNE 2D", fontsize=15)
            sns.scatterplot(
                x="tsne-2d-1", y="tsne-2d-2",
                hue=true_label_col,
                palette=sns.color_palette("Set2", len(labels)),
                data=self.df,
                legend="full", ax=ax1)
            
            ax2 = fig.add_subplot(2,2,2, projection='3d')
            ax2.set_title("T-SNE 3D", fontsize=15)
            ax2.set_xlabel('tsne-3d-1')
            ax2.set_ylabel('tsne-3d-2')
            ax2.set_zlabel('tsne-3d-3')
            for lab in labels:
                xs = self.df['tsne-3d-1'][self.df[true_label_col]==lab]
                ys = self.df['tsne-3d-2'][self.df[true_label_col]==lab]
                zs = self.df['tsne-3d-3'][self.df[true_label_col]==lab]
                ax2.scatter(xs, ys, zs, label=lab)
            ax2.legend()

            ax3 = fig.add_subplot(2,2,3)
            ax3.set_title('PCA 2D', fontsize=15)
            sns.scatterplot(
                    x="PC1", y="PC2",
                    hue=true_label_col,
                    palette=sns.color_palette("Set2", len(labels)),
                    data=self.df,
                    legend="full", ax=ax3)

            ax4 = fig.add_subplot(2,2,4, projection='3d')
            ax4.set_title("PCA 3D", fontsize=15)
            ax4.set_xlabel('PC1')
            ax4.set_ylabel('PC2')
            ax4.set_zlabel('PC3')
            for lab in labels:
                xs = self.df['PC1'][self.df[true_label_col]==lab]
                ys = self.df['PC2'][self.df[true_label_col]==lab]
                zs = self.df['PC3'][self.df[true_label_col]==lab]
                ax4.scatter(xs, ys, zs, label=lab)
            ax4.legend()

        if save_plot_path is not None:
            fig.savefig(save_plot_path)
            
        return self.df


    @staticmethod
    def get_corpus_cluster_df(corpus: list, sentence_transformer_model_name: str, num_cluster: int) -> list:
        """using the corpus list as input, clustered information is input as a prefix to each document to return the entire clusterd corpus list.

        :param corpus: corpus list (1d list)
        :type corpus: list
        :param sentence_transformer_model_name: model name of Sentence Transformer (see readme.md)
        :type sentence_transformer_model_name: str
        :param num_cluster: number to cluster
        :type num_cluster: int
        :return: a corpus list containing clustered information as a prefix (1d list)
        :rtype: list
        """
        embedder = SentenceTransformer(sentence_transformer_model_name)
        corpus_embeddings = embedder.encode(corpus)
        k_means_model = KMeans(n_clusters=num_cluster)
        k_means_model.fit(corpus_embeddings)
        cluster_assignment = k_means_model.labels_
        clustered_corpus = [[] for i in range(num_cluster)]
        for sentence_id, cluster_id in tqdm(enumerate(cluster_assignment), total=len(corpus)):
            clustered_corpus[cluster_id].append(corpus[sentence_id])
        clustered_list = [[str(i)+'cluster'+x for x in clustered_corpus[i]] for i in range(num_cluster)]

        return sum(clustered_list, [])


    def get_df_cluster(self) -> pd.DataFrame:
        """create a column containing clustered information in a data frame

        :return: a data frame with information clustered in the 'cluster' column
        :rtype: pd.DataFrame
        """
        self.df.reset_index()
        self.df[self.target_col] = [f'{i}index{b}' for i, b in enumerate(self.df[self.target_col])] 
        corpus = self.df[self.target_col].to_list()
        clusted_corpus = self.get_corpus_cluster_df(corpus, self.sentence_transformer_model_name, self.num_cluster)
        
        df_cluster = pd.DataFrame(clusted_corpus, columns=[self.target_col])
        df_cluster['cluster'] = [int(x.split('cluster')[0]) for x in df_cluster[self.target_col]]
        df_cluster[self.target_col] = [x.split('cluster')[1] for x in df_cluster[self.target_col]]
        df_cluster['index'] = [int(x.split('index')[0]) for x in df_cluster[self.target_col]]
        df_cluster[self.target_col] = [x.split('index')[1] for x in df_cluster[self.target_col]]
        df_cluster.drop([self.target_col], axis = 1, inplace=True)

        self.df['index'] = [int(x.split('index')[0]) for x in self.df[self.target_col]]
        self.df[self.target_col] = [x.split('index')[1] for x in self.df[self.target_col]]
        self.df = self.df.join(df_cluster.set_index('index'), on='index')

        return self.df
    

    def embed(self, x: str) -> np.array:
        """perform embedding

        :param x: document to embed
        :type x: str
        :return: embedded array
        :rtype: np.array
        """
        return self.embedder.encode(str(x))


    def quick_cluster_show(self, vis_type: str, show_off: bool, save_plot_path: str) -> pd.DataFrame:
        """visualize clustering information using labels

        :param vis_type: 'tsne2d', 'tsne3d', 'pca2d', 'pca3d', 'joint_pca2d', 'joint_tsne2d'
        :type vis_type: str
        :param show_off: plot will not be visible when entering 'True'
        :type show_off: bool
        :param save_plot_path: plot will not be saved when None is entered
        :type save_plot_path: str
        :return: a data frame with information clustered in the 'cluster' column and has back-data columns for plotting
        :rtype: pd.DataFrame
        """
        if 'cluster' not in self.df.columns:
            self.df = self.get_df_cluster()
        if 'embedded_sentence' not in self.df.columns:
            self.df['embedded_sentence'] = [self.embed(x) for x in tqdm(self.df[self.target_col])]

        if vis_type == 'tsne2d':
            return_df = qs.vis_tsne2d(self.df, 'embedded_sentence', 'cluster', show_off, save_plot_path)
        elif vis_type == 'tsne3d':
            return_df  = qs.vis_tsne3d(self.df, 'embedded_sentence', 'cluster', show_off, save_plot_path)
        elif vis_type == 'pca2d':
            return_df = qs.vis_pca(self.df, 'embedded_sentence', 'cluster', 2, show_off, save_plot_path)
        elif vis_type == 'pca3d':
            return_df = qs.vis_pca(self.df, 'embedded_sentence', 'cluster', 3, show_off, save_plot_path)
        elif vis_type == 'joint_pca2d':
            return_df = qs.joint_pca2d(self.df, 'embedded_sentence', 'cluster', 2, show_off, save_plot_path)
        elif vis_type == 'joint_tsne2d':
            return_df = qs.joint_tsne2d(self.df, 'embedded_sentence', 'cluster', show_off, save_plot_path)
        
        return return_df


    def quick_corpus_show(self, true_label_col: str, vis_type: str, show_off: bool, save_plot_path: str) -> pd.DataFrame:
        """if real labels exist, create plot with real labels

        :param true_label_col: actual label column name
        :type true_label_col: str
        :param vis_type: 'tsne2d', 'tsne3d', 'pca2d', 'pca3d', 'joint_pca2d', 'joint_tsne2d'
        :type vis_type: str
        :param show_off: plot will not be visible when entering 'True'
        :type show_off: bool
        :param save_plot_path:  plot will not be saved when None is entered
        :type save_plot_path: str
        :return: Dataframe with back data for plotting
        :rtype: pd.DataFrame
        """
        if 'embedded_sentence' not in self.df.columns:
            self.df['embedded_sentence'] = [self.embed(x)for x in tqdm(self.df[self.target_col])]

        if vis_type == 'tsne2d':
            return_df = qs.vis_tsne2d(self.df, 'embedded_sentence', true_label_col, show_off, save_plot_path)
        elif vis_type == 'tsne3d':
            return_df  = qs.vis_tsne3d(self.df, 'embedded_sentence', true_label_col, show_off, save_plot_path)
        elif vis_type == 'pca2d':
            return_df = qs.vis_pca(self.df, 'embedded_sentence', true_label_col, 2, show_off, save_plot_path)
        elif vis_type == 'pca3d':
            return_df = qs.vis_pca(self.df, 'embedded_sentence', true_label_col, 3, show_off, save_plot_path)
        elif vis_type == 'joint_pca2d':
            return_df = qs.joint_pca2d(self.df, 'embedded_sentence', true_label_col, 2, show_off, save_plot_path)
        elif vis_type == 'joint_tsne2d':
            return_df = qs.joint_tsne2d(self.df, 'embedded_sentence', true_label_col, show_off, save_plot_path)
        
        return return_df


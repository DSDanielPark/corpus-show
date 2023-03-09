import pandas as pd
import quickshow as qs
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


class CorpusCluster:
    def __init__(self, csv_file_path: str, sentence_transformer_model_name: str, target_col: str, num_cluster: int) -> None:
        if csv_file_path is not None:
            self.df = pd.read_csv(csv_file_path)
        if sentence_transformer_model_name is None:
            sentence_transformer_model_name = 'paraphrase-xlm-r-multilingual-v1'
        self.embedder =  SentenceTransformer(sentence_transformer_model_name)
        self.num_cluster = num_cluster
        self.sentence_transformer_model_name = sentence_transformer_model_name
        self.target_col = target_col


    @staticmethod
    def get_corpus_cluster_df(corpus: list, sentence_transformer_model_name: str, num_cluster: int) -> list:
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
    

    def embed(self, x):
        return self.embedder.encode(str(x))


    def quick_cluster_show(self, vis_type: str, show_off: bool, save_plot_path: str) -> pd.DataFrame:
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
        
        return return_df


    def quick_corpus_show(self, true_label_col: str, vis_type: str, show_off: bool, save_plot_path: str) -> None:
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
        
        return return_df


    def show(self, true_label_col: str, save_plot_path: str) -> pd.DataFrame:
        if true_label_col == 'cluster':
            if 'cluster' not in self.df.columns:
                self.df = self.get_df_cluster()
            if 'embedded_sentence' not in self.df.columns:
                self.df['embedded_sentence'] = [self.embed(x) for x in tqdm(self.df[self.target_col])]
            qs.vis_tsne2d(self.df, 'embedded_sentence', 'cluster', True, None)
            qs.vis_tsne3d(self.df, 'embedded_sentence', 'cluster', True, None)
            qs.vis_pca(self.df, 'embedded_sentence', 'cluster', 3, True, None)

            fig = plt.figure(figsize=plt.figaspect(1))
            ax = fig.add_subplot(2,2,2, projection='3d')
            x, y, z = self.df['tsne-3d-1'], self.df['tsne-3d-2'], self.df['tsne-3d-3']
            ax.scatter3D(x, y, z)
            plt.title("T-SNE 3D", fontsize=15)
            for s in self.df['cluster'].unique():
                ax.scatter(self.df['tsne-3d-1'][self.df['cluster']==s], 
                        self.df['tsne-3d-2'][self.df['cluster']==s], 
                        self.df['tsne-3d-3'][self.df['cluster']==s], 
                        label=s)
            ax.set_xlabel('tsne-3d-1')
            ax.set_ylabel('tsne-3d-2')
            ax.set_zlabel('tsne-3d-3')
            ax.legend()

            plt.subplot(2,2,3)
            plt.title('PCA 2D', fontsize=15)
            sns.scatterplot(
                    x="PC1", y="PC2",
                    hue='cluster',
                    palette=sns.color_palette("Set2", len(self.df['cluster'].unique())),
                    data=self.df,
                    legend="full"
                    )

            plt.subplot(2,2,1)
            plt.title("T-SNE 2D", fontsize=15)
            sns.scatterplot(
                x="tsne-2d-1", y="tsne-2d-2",
                hue='cluster',
                palette=sns.color_palette("Set2", len(self.df['cluster'].unique())),
                data=self.df,
                legend="full"
                )

            ax = fig.add_subplot(2,2,4, projection='3d')
            plt.title("PCA 3D", fontsize=15)
            fig = plt.figure(figsize=plt.figaspect(1))
            for s in self.df['cluster'].unique():
                ax.scatter(self.df['PC1'][self.df['cluster']==s], 
                        self.df['PC2'][self.df['cluster']==s], 
                        self.df['PC3'][self.df['cluster']==s], 
                        label=s)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            ax.legend()

            if save_plot_path is not None:
                plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')

            plt.show()
        
        else:
            if 'embedded_sentence' not in self.df.columns:
                self.df['embedded_sentence'] = [self.embed(x)for x in tqdm(self.df[self.target_col])]
            self.df = qs.vis_tsne2d(self.df, 'embedded_sentence', true_label_col, True, None)
            self.df = qs.vis_tsne3d(self.df, 'embedded_sentence', true_label_col, True, None)
            self.df = qs.vis_pca(self.df, 'embedded_sentence', true_label_col, 3, True, None)

            fig = plt.figure(figsize=plt.figaspect(1))
            ax = fig.add_subplot(2,2,2, projection='3d')
            x, y, z = self.df['tsne-3d-1'], self.df['tsne-3d-2'], self.df['tsne-3d-3']
            ax.scatter3D(x, y, z)
            plt.title("T-SNE 3D", fontsize=15)
            for s in self.df[true_label_col].unique():
                ax.scatter(self.df['tsne-3d-1'][self.df[true_label_col]==s], 
                        self.df['tsne-3d-2'][self.df[true_label_col]==s], 
                        self.df['tsne-3d-3'][self.df[true_label_col]==s], 
                        label=s)
            ax.set_xlabel('tsne-3d-1')
            ax.set_ylabel('tsne-3d-2')
            ax.set_zlabel('tsne-3d-3')
            ax.legend()

            plt.subplot(2,2,3)
            plt.title('PCA 2D', fontsize=15)
            sns.scatterplot(
                    x="PC1", y="PC2",
                    hue=true_label_col,
                    palette=sns.color_palette("Set2", len(self.df[true_label_col].unique())),
                    data=self.df,
                    legend="full"
                    )

            plt.subplot(2,2,1)
            plt.title("T-SNE 2D", fontsize=15)
            sns.scatterplot(
                x="tsne-2d-1", y="tsne-2d-2",
                hue=true_label_col,
                palette=sns.color_palette("Set2", len(self.df[true_label_col].unique())),
                data=self.df,
                legend="full"
                )

            ax = fig.add_subplot(2,2,4, projection='3d')
            plt.title("PCA 3D", fontsize=15)
            fig = plt.figure(figsize=plt.figaspect(1))
            for s in self.df[true_label_col].unique():
                ax.scatter(self.df['PC1'][self.df[true_label_col]==s], 
                        self.df['PC2'][self.df[true_label_col]==s], 
                        self.df['PC3'][self.df[true_label_col]==s], 
                        label=s)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            ax.legend()

            if save_plot_path is not None:
                plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')

            plt.show()

        
            
        return self.df
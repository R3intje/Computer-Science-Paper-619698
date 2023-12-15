# Computer-Science-Paper-619698
Scalable Product Duplicate Detection

Online web shopping has become increasingly popular in the past decades. Due
to the rise of access to the internet, more and more people can buy products
online. There are several reasons why someone might be interested in knowing
what products are available on the online market. Consumers want to make
informed purchasing decisions based on comparing the product’s features and
price. But also businesses need to be aware of what products are in their industry
to stay competitive. Unfortunately, it is not an easy task to get an overview of
all products on the market. Most products are represented differently per online
webshop. It is therefore a tedious task to find product duplicates manually. For
this reason, an algorithm is built to detect duplicate pairs in a time-efficient
manner.
This code introduces a new set of Model Words to represent the product’s
title. The brand of the television will be added to the set of model words of that
product.

1. The dataset containing 1624 products is loaded. The shop, ModelID, featuresMap, title and URL are given per product.
2. The data is cleaned by removing the URL column and adding the Brands column.
3. Model Words of the titles are extracted.
4. The set of all model words is created, now the set of model words per title can be computed.
5. Real duplicate pairs are found using identify_duplicate_pairs(df).
6. The binary matrix is created using generate_binary_vectors(MW_title, title, MW).
7. The signature matrix is generated out of the binary matrix using minHash(binary_matrix_np, num_minhashes).
8. Now it's time for LSH: For different r_values, the metrics are calculated using calculate_metrics_for_r_values(r_values, signature_matrix, ID_duplicates, N, df, MW_title).
   repeat for every r = [2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 25, 30, 40, 48, 50, 60, 75, 80, 100, 120, 150, 200, 240, 300, 400, 600]
     8a. First the candidate pairs are extracted locality_sensitive_hashing(signature_matrix, max(1, int(signature_matrix.shape[0] // r)), r).
     8b. The dissimilarity matrix is generated using dissimilarity_matrix(candidate_pairs, df, MW_title).
     8c. The MSM duplicates are found using linkage_clustering(diss_matrix, threshold) with theshold = 0.5.
     8d. The LSH metrics are calculated using calculate_metrics(ID_duplicates, candidate_pairs_set, N).
     8e. The F1 MSM metric is calculated using calculate_f1(ID_duplicates, duplicate_pairs_set).
9. The metrics can now be plotted using plot_metrics(metrics_results, 'Metric plots').

10. Run step 1-9 multiple times with bootstrapping as follows: metrics_results = run_algorithm(df, nBootstraps = 1).

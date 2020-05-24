# Bayesian Recommendation Systems using MCMC Methods

A Collaborative Filtering based movie recommendation system is built below by recommending the top n movies for a user based on the past movie ratings of the user. Bayesian Hierarchical models are used to parameterize each user and movie using a 10-dimensional vector. Gibbs sampling is used to predict the user ratings for all movies and recommend the best n movies for any user based on their past movie ratings.

 This can done by 

• Accurately predicting the ratings for all the movies for the user and returning the movies with the top n predicted ratings 

• Compute the ratings using the dot product between the user vector and movie vector

• Sample the user and movie vectors from the joint posterior of the Bayesian Hierarchical Model using Gibbs Sampling

# Advantages:

Many movie recommendation engines including IMDb use the similarity of movies using content based features like actors, genre etc. or collaborative ﬁltering to recommend the movie groups which are similar to a given movie where the input usually is just one movie. Users typically have more complicated usage patterns and recommending movies which are very similar or which are of the same genre may not be an effective strategy.

kNN clustering and Matrix factorization methods are some of the very commonly used Collaborative Filtering(CF) methods. Matrix Factorization is usually done using Singular Value Decomposition(SVD), but models using SVD are very prone to overﬁtting. This is particularly relevant to this problem as many users and movies have only few ratings across all possible values as the ratings matrix is very sparse due to the huge number of users and movies.

The Clustering methods usually group movies and users into speciﬁc groups like movie Genres, User groups who only watch Comedy movies etc.. These groups are optimized for maximum information content when used in an SVD setting and can have poor performances. For eg. some users might just watch movies of a speciﬁc actor or director. We would ideally want a system which considers each and every user and movie individually as opposed to parts of a group or a big cluster.

Since Matrix factorization is a very computationally intensive process, it can be performed only on the complete matrix in ﬁxed time intervals. This might lead to delay in capturing patterns and giving the latest recommendations for a new user or new movie.
A Bayesian Hierarchical model is proposed here to address the problems mentioned above. It can be seen as a regularized approximation to the Matrix Factorization method as it uses a zero prior and a dot product between the User vector and Movie vector to approximate the rating. It is also more ﬂexible from an online learning perspective to update the user vectors on the ﬂy. The movie recommendation engine considered here analyzes the movie ratings provided by each user for each movie and models the user features and movie features using a k-dimensional vector. These user and movie vectors can be used to extrapolate the ratings across all movies which were not seen by the user and recommend the ones which the user is most likely to rate the highest.

# Method

The data was collected from the Full MovieLens and TMDb datasets available on Kaggle. The Full MovieLens Dataset contains metadata for 45,000 movies released on or before July 2017. Data points include cast, crew, plot keywords, budget, revenue, posters, release dates, languages, production companies, countries, TMDB vote counts and vote averages. The metadata is primarily used for ﬁnal analysis of the quality of recommendations.

The data from ofﬁcial GroupLens website contains 26 million ratings from 270,000 users for the 45,000 movies. Ratings are on a scale of 1-5. The user data is anonymized, but the movie IDs can be used to fetch the real movie name and descriptions from the metadata above.
For this project we use a smaller section of the full ratings dataset due to computational limitations. The smaller dataset used consists of 671 users and 9066 movies. 


The Bayesian Hierarchical model we use is hown below. 

![Hierarchical Model](https://github.com/Arunachalam-M/Bayesian-Recommendation-Systems/blob/master/hier_model.JPG)

Our Parameters and hyperparameters are as below.

![Hierarchical Model](https://github.com/Arunachalam-M/Bayesian-Recommendation-Systems/blob/master/2.JPG)


Using the above model, we can derive the conditional distributions of all the parameters as below for Gibbs Sampling.

![Gibbs Sampling](https://github.com/Arunachalam-M/Bayesian-Recommendation-Systems/blob/master/3.JPG)

The above distributions are used for Gibbs Sampling to sample from the joint posterior distribution. After we have all the User and Movie vectors, we can obtain the pairwise prediction matrix for all User, Movie combinations simply using the dot product of the User and Movie Vectors.

# Implementation

The above recommendation engine is implemented in python using NumPy, pandas, SciPy and was run in Google Colab for 10000 iterations over a period of 3 days. Through efﬁcient implementation, each iteration takes about 2.4 seconds. The User and Movie vectors are each a m x 10 and n x 10 matrix for each iteration. These are too expensive to store and are discarded after the ratings are computed. Python was chosen over R for the following reasons: 

• The dataset needs multiple joins between different tables which are easier and more efﬁcient in pandas 

• Python data structures like dictionaries are very efﬁcient for mapping of movies and users with their Ids 

• The operations are vectorized for faster implementation using NumPy 

• SciPy sparse matrices are used to store and compute the full matrix which cannot be handled by RAM otherwise 

• To mitigate crashes during longer runs of the code, backups are auto written on google drive from google colab

# Results

The predicted ratings ﬁle is downloaded from the mounted drive and further analysis is done. Since the user data is anonymized, we merge the movie IDs with the movies metadata to ﬁnd the top 10 movies rated by the user and the top 10 recommendations from the system to compare the relevance of the recommendations. Since the subset is very small, it has fewer popular movies and many movies from mid 20th century among it. Following are some of the original ratings and recommendations by the system.

User 121: 

Seen : [’Strange Days’, ’Eyes Wide Shut’, ’Lili Marleen’, ’To Be or Not to Be’, ’Three Colors: Red’, ’The Dark’, ’Aliens vs Predator: Requiem’, ’Terminator 3: Rise of the Machines’, ’Notorious’, ’A River Runs Through It’] 

Recommended : [’Superman’, ’Earth’, ’To Be or Not to Be’, ’Land of Plenty’, ”Carla’s Song”, ’The Whole Ten Yards’, ”Wayne’s World 2”, ’High and Dizzy’, ’The Coast Guard’, ’Citizen Kane’]

User 556: 

Seen : [’48 Hrs.’, ’The Hours’, ”Ocean’s Eleven”, ’Cockles and Muscles’, ’Romeo + Juliet’, ’Sissi’, ’Back to the Future Part II’, ’Monsoon Wedding’, ’Solaris’, ’Grill Point’] 

Recommended : [’Three Colors: Red’, ’Sissi’, ’Who Killed Bambi?’, ’Monsoon Wedding’, ”Dave Chappelle’s Block Party”, ’8 Mile’, ’Grill Point’, ’The Next Best Thing’, ’The Passion of Joan of Arc’, ’Star Trek IV: The Voyage Home’]

User 519: 

Seen : [’License to Wed’, ’The Butterﬂy Effect’, ’The Demolitionist’, ’Dawn of the Dead’, ’The Prisoner of Zenda’, ’Local Color’, ’Cousin, Cousine’, ’The Talented Mr. Ripley’, ’Under the Sand’, ’Point Break’] 

Recommended : [’Solaris’, ’Monsoon Wedding’, ’Sissi’, ’A Nightmare on Elm Street’, ’Terminator 3: Rise of the Machines’, ’Batman Returns’, ’The Million Dollar Hotel’, ’Young and Innocent’, ’Three Colors: Red’, ’Once Were Warriors’]


From the above recommendations, we can see that the recommended movies are both popular and similar to the movies seen by the user. Some users got some movies recommended which are in their top rated list. We can also notice that the movies recommended are signiﬁcantly different for each user. To see if they are line with the recommendations, we plot the mean of predicted ratings vs actual ratings of the movies already rated i.e. the training data. The following graph shows the range of predicted ratings for each actual rating across users.

![Results: Actual vs Predicted](https://github.com/Arunachalam-M/Bayesian-Recommendation-Systems/blob/master/predictions.png)

Here we can clearly see that the average predicted ratings for a movie by the recommendation system is proportional to the actual rating by the user. With more iterations and more data, the variance in the predicted ratings can be reduced to be more accurate. The current model has captured the patterns and has provided good recommendations without overﬁtting despite very low data for some users and the sparseness of the matrix. 








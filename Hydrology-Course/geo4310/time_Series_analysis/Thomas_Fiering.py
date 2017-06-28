
def thomas_fiering(q_hist, q_init, j_init, period):
    """ Thomas Fiering Model """
    
    q = q_hist

    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    months1 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1]
    
    # create a random generator, note we have to call this '()' and take the
    # first element returned [0]... so in use it is Z()[0]
    Z = np.random.randn

    # get the mean monthly discharges for the j months
    q_mean_j = [q[q.index.month == i].discharge.mean() for i in months]
        
    # get the mean monthly discharges for the j+1 months
    q_mean_j1 = [q[q.index.month == i].discharge.mean() for i in months1]
    
    # do the same for standard deviation
    sj = [q[q.index.month == i].discharge.std() for i in months]
        
    sj1 = [q[q.index.month == i].discharge.std() for i in months1]
    
    # calculate the pearson r using built-ins
    rj = [pearsonr(q[q.index.month == i].discharge,
                   q[q.index.month == j].discharge)[0] for i, j in zip(months, months1)]

    # the model, not vectorized. assumes each of the variables above are
    # of length 12 ... and note that Python is zero-indexed (e.g. first element is '0th') 
    def tf1(qi, month):
        i = month-1 #account for zero offset
        return q_mean_j1[i] + (rj[i] * (sj1[i] / sj[i])) * (qi - q_mean_j[i]) + Z(1)[0] * sj1[i] * np.sqrt((1 - rj[i] ** 2))
        
    # initiate simulated q with q_init and j_init (month)
    # simq is a list, and we just call the model function to get the first element
    simq = [tf1(q_init, j_init)]

    # enumerate (e.g. makes 'i' a counter) and get each month in the simulation period
    for i, m in enumerate(period.month[1:]):

        #append to the list by calling the model with the prior value and the month
        simq.append(tf1(simq[i-1], m))

    # convert it to a dataframe for convenience
    return pd.DataFrame(np.array(simq), columns=['discharge'], index=period)

#if __name__ == "__main__": 

# MAL
Active Learning using open source scikit tools 

At this stage hard coded to use certain news data files. Use this component iteratively.

##Usage

    from SimpleAL import SimpleAL
    AL=SimpleAL()
    AL.load_pennystocktweet_news()
    AL.run_classifier(init=True)
    print AL.get_next()
    #Real news
    AL.set_label(-1.0)
    #Suspicious news
    #AL.set_label(1.0)
    AL.run_classifier()
    print AL.get_next()


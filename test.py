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

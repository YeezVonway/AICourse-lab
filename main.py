import audio_classification as ac

if __name__ == "__main__":

   
    ac.CONFIG.GLOBAL.TRAIN.LR = 3e-7


    # ac.CONFIG.GLOBAL.DATA.CLASSES =[('cough', 5.6568627450980395), ('default', 1.4485355648535565), ('scream', 17.31 + 5), ('sneeze', 13.315384615384616 + 4)]

    ac.CONFIG.commit()
    
    ac.train.train(rnd=500, logFile='./trainlog.log')

    ac.CONFIG.GLOBAL.DATA.ARGUMENTATION = None
    ac.test.test()
    
    
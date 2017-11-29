
import myHMM  
import numpy as np
import sys

if sys.version_info.major == 3:
    xrange = range

hmm = myHMM.myHMM()


test_start_date = '2017-01-01'
test_end_date = '2017-11-11'
all_data = hist_prices = myHMM.parseStockPrices(test_start_date, test_end_date, '002415')
 


num_correct=0.0
test_window = 6
N=len(all_data)
num_tests=N//test_window
for n in xrange(1,N-test_window,test_window):
    train_data = all_data[-n:-n-test_window:-1,:]
    hist_moves = myHMM.calculateDailyMoves(train_data,1)
    hist_O=np.array(list(map(lambda x: 1 if x>0 else (0 if x<0 else 2), hist_moves)))
    hist_O=hist_O[::-1]
    (a, b, pi_est, alpha_est) = hmm.HMMBaumWelch(hist_O, 2, False, False)
    path = hmm.HMMViterbi(a, b, hist_O, pi_est)
    prediction_state=np.argmax(a[int(path[-1]),:])
    prediction = np.argmax(b[prediction_state,:])
    if ((all_data[-n-test_window-1,1]-all_data[-n-test_window,1])>0 and prediction==1) or ((all_data[-n-test_window-1,1]-all_data[-n-test_window,1])<0 and prediction==0) or ((all_data[-n-test_window-1,1]-all_data[-n-test_window,1])==0 and prediction==2):
        num_correct+=1.0
print(num_correct/num_tests)

import tensorflow as tf

class lstm_model():
    def __init__(self):        
        return

    def output(self, x, batch_size, state_size, num_layers, num_steps_total, num_steps_autoregress):
        """Creates an LSTM model
        :param x: input Tensor of shape batch_size x num_steps x dimensions
        
        :param state_size: size of hidden state of LSTM, integer
        :param num_layers: number of recurrent unit layers, integer
        :param num_steps_total: total number of time steps over which recurrent unit is run, integer
        :param num_steps_auto_regress: steps for which lstm input is previous output
        """
        
        stacked_cell = []
        for i in range(num_layers):
            stacked_cell.append(tf.nn.rnn_cell.LSTMCell(state_size))
        
        cell = tf.nn.rnn_cell.MultiRNNCell(stacked_cell, state_is_tuple=True)
            
        output = [None]*num_steps
        state =  cell.zero_state(batch_size, tf.float32)
       
        for t in range(num_steps_total):
            x_t = x[:,t,:]         
            output[t], state  = cell(x_t, state)
        
        return output, state
        
def make_lstm_cell(batch_size, state_size, num_layers, reuse):
    
    stacked_cell = []
    for i in range(num_layers):
        stacked_cell.append(tf.nn.rnn_cell.LSTMCell(state_size))
    cell = tf.nn.rnn_cell.MultiRNNCell(stacked_cell, state_is_tuple=True)
    init_state = cell.zero_state(batch_size, tf.float32)
    return cell, init_state


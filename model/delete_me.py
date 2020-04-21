import tensorflow as tf

@tf.RegisterGradient("CustomClipGrad")
def _clip_grad(unused_op, grad):
  return tf.scalar_mul(2, tf.clip_by_value(grad, -0.1, 0.1))

input = tf.Variable([3.0], dtype=tf.float32)

# output without gradient clipping in the backwards pass for comparison:
output = tf.multiply(input,input)
grad = tf.gradients(output, input)

g = tf.get_default_graph()
with g.gradient_override_map({"Identity": "CustomClipGrad"}):
  output_clip = tf.identity(input, name="Identity")
grad_clip = tf.gradients(output_clip, input)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print("with clipping:", sess.run(grad_clip)[0])
  print("without clipping:", sess.run(grad)[0])
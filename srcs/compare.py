from srcs.train import load_history, plot_multiple_curves

h1 = load_history('models/model_32_history.json')
h2 = load_history('models/model_128_history.json')

plot_multiple_curves(
    [h1, h2],
    labels=['32x32', '128x128'],
    save_path='outputs/comparison.png'
)

from opts import parse_opts
from model import generate_model
opt = parse_opts()
model, parameters = generate_model(opt)
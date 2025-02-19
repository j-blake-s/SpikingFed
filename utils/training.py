

import torch
def train(model, data, opt, err, pred, args):
  model.train()
  total_samples = 0
  loss_sum = 0
  total_correct_samples = 0
  for i, (images, labels) in enumerate(data):

    # Run Model #
    images = images.to(args.device)
    images = images.to(torch.float32)
    labels = labels.to(args.device)
    labels = labels.to(torch.int64)    
    outputs = model(images)

    # Loss #
    loss = err(outputs, labels)
    loss.backward()
    opt.step()
    opt.zero_grad()


    # Stats #
    total_samples += images.shape[0]
    loss_sum += loss.cpu().data.item() * outputs.shape[0]
    correct_samples = torch.sum(pred(outputs)==labels).cpu().data.item()
    total_correct_samples += correct_samples

    # Print Stats #
    acc = total_correct_samples / total_samples

    print(f'\r\tBatch [{i+1}/{len(data)}] Training: {acc:.2%}',end="")

  return acc

def test(model, data, pred, args):
  model.eval()
  total_samples = 0
  total_correct_samples = 0
  with torch.no_grad():
    for i, (images, labels) in enumerate(data):
      
      # Run Model #
      images = images.to(args.device)
      images = images.to(torch.float32)
      labels = labels.to(args.device)
      outputs = model(images)

      # Stats #
      total_samples += images.shape[0]



      correct_samples = torch.sum(pred(outputs)==labels).cpu().data.item()
      total_correct_samples += correct_samples

      # Print Stats #
      acc = total_correct_samples / total_samples

      print(f'\r\tBatch [{i+1}/{len(data)}] Validation: {acc:.2%}',end="")

  return acc
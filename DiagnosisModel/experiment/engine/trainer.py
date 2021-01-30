import time


class trainer():

    def __init__(self, model, dataloader, optimizer, scheduler, logger, log_per_step):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.log_per_step = log_per_step

    def __call__(self, epoch):
        self.model.train()

        running_loss_dict = {}
        start_t = time.time()
        for i, data_dict in enumerate(self.dataloader):

            # 1. forward
            loss_dict = self.model(data_dict)

            # 2. update weights
            self.optimizer.zero_grad()
            loss_dict['total'].mean().backward()
            self.optimizer.step()

            # 3. update running loss
            for k, v in loss_dict.items():
                if k not in running_loss_dict.keys():
                    running_loss_dict[k] = 0.0
                running_loss_dict[k] += v.mean().item()

            # 4. logger       
            if i % self.log_per_step == self.log_per_step-1:
                # epoch & step
                train_string = '[Epoch {:0>3} Step {:0>3}|{:0>3}] '.format(epoch+1, i+1, len(self.dataloader))
                # loss
                train_string += 'loss: '
                for k, v in running_loss_dict.items():
                    train_string += '[{} {:.4f}] '.format(k, v/self.log_per_step)
                # time
                train_string += 'time: {:.2f}s'.format(time.time()-start_t)
                print(train_string)
                # tensorboard
                for k, v in running_loss_dict.items():
                    self.logger.add_scalar(f'loss {k}', v/self.log_per_step, i+epoch*len(self.dataloader))
                # reinitialization
                running_loss_dict = {}
                start_t = time.time()

        self.logger.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)

        if self.scheduler is not None: self.scheduler.step()
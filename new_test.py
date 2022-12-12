from multiprocessing import Process, Manager, Queue


# class haha():
#     def __init__(self):
#         self.a = ["label"]
#         self.run()
#     def test(self, image, a):
#         a.put(image)
#
#     def run(self):
#         q = Queue()
#         jobs = []
#         for i in ['original', 'log-sigma', 'wave']:
#             p = Process(target=self.test, args=([i], q))
#             p.start()
#             jobs.append(p)
#         for p in jobs:
#             p.join()
#         haha = [q.get() for j in jobs]
#         for i in haha:
#             self.a += i
#             print(self.a)
        # for i, p in zip(['original', 'log-sigma', 'wave'], jobs):
        #     p.join()
        #     b = q.get()
        #     self.a += b
        #     print(b)
        #     if i != 'original':
        #         print(self.a)
if __name__ == "__main__":
    image_types = ['original', 'log-sigma', 'wave']
    feature= [1,2,3]
    for i, j in zip(image_types, feature):
        print(i)
    for i, j in zip(image_types, feature):
        print(i)
if __name__ == "__main__":

    if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath', default='./data',
                        help="data path")
    parser.add_argument('--noTraining', default='10000',
                        help="number of training examples")
    parser.add_argument('--noValidation', default='1000',
                        help="number of validation examples")
    parser.add_argument('--noTesting', default='1000',
                        help="number of testing examples")
    parser.add_argument('--batchSize', default='64',
                        help="batch size")
    parser.add_argument('--learningRate', default='1e-4',
                        help="initial learning rate")

    args = parser.parse_args()

    args.id += '--batchSize' + str(args.batchSize)
    args.id += '--learningRate' + str(args.learningRate)
    
    main()
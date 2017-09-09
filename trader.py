class trader:

    def __init__(self, balance):
        self.balance = balance

    def margin(self, buy, sell, amount, price, aofsell= False, pofsell=False):
        if pofsell:
            price = 1/price

        if aofsell:
            amount = amount/price

        b = self.balance[buy]
        s = self.balance[sell]

        b += amount
        s -= amount*price

        if s<0:
            b += s/price
            s = 0

        self.balance[buy] = b
        self.balance[sell] = s

    def getBalance(self, currency=None):
        if currency:
            return self.balance[currency]
        else:
            return self.balance

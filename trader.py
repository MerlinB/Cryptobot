class trader:

    balance = {}
    
    def __init__(self, balance):
        self.balance = balance

    def margin(self, buy, sell, amount, price, aofsell= False, pofsell=False):
        if pofsell:
            price = 1/price

        if aofsell:
            amount = amount/price
            
        if not buy in self.balance:
            self.balance[buy] = 0
        if not sell in self.balance:
            self.balance[sell] = 0
            
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

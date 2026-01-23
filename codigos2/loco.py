class Persona:

    def _init_(self, pNombre, pEdad):
        self._nombre = pNombre
        self._edad = pEdad

    def mostrar(self):
        print("Este es el método mostrar")

    def getNombre(self):
        return self._nombre

    def setNombre(self, newNombre):
        self._nombre = newNombre

    def getEdad(self):
        return self._edad

    def setEdad(self, newEdad):
        self._edad = newEdad


class Empleado(Persona):

    def _init_(self, pNombre, pEdad, pSueldoBruto):
        super()._init_(pNombre, pEdad)
        self._sueldoBruto = pSueldoBruto

    def mostrar(self):
        return super().mostrar()

    def calcularSalarioNeto(self):
        print("Este es el método calcularSalarioNeto")

    def getSueldoBruto(self):
        return self._sueldoBruto

    def setSueldoBruto(self, newSueldoBruto):
        self._sueldoBruto = newSueldoBruto


class Cliente(Persona):

    def _init_(self, pNombre, pEdad, pEmpresa, pTelefono):
        super()._init_(pNombre, pEdad)
        self._empresa = pEmpresa
        self._telefono = pTelefono

    def mostrar(self):
        return super().mostrar()

    def getEmpresa(self):
        return self._empresa

    def setEmpresa(self, newEmpresa):
        self._empresa = newEmpresa

    def getTelefono(self):
        return self._telefono

    def setTelefono(self, newTelefono):
        self._telefono = newTelefono


class Empresa:

    def _init_(self, pNombre, pCliente):
        self.__nombre = pNombre
        self.__cliente = pCliente

    def getNombre(self):
        return self.__nombre

    def setNombre(self, newNombre):
        self.__nombre = newNombre

    def getCliente(self):
        return self.__cliente

    def setCliente(self, newCliente):
        self.__cliente = newCliente
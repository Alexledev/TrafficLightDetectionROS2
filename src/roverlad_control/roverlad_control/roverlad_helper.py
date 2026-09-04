class RoverladPIDDrive:
    def __init__(self, linearVel, angularVel):
        self.kp = 0.02
        self.ki = 0.0
        self.kd = 0.01
        
        # self.currAngle = 0.0
        self.integral = 0.0
        self.prevError = 0.0

        self.maxAngular = 1.5

        self.linearVel = linearVel        
        self.angularVel = angularVel

    def setLinearAngularVel(self, lin, ang):        
        self.linearVel = lin        
        self.angularVel = ang

    def PIDCalc(self, error):
       
        self.integral = self.clamp(self.integral + error, -1000, 1000)

        derivative = error - self.prevError
        self.prevError = error

        angular = (self.kp * error + self.ki * self.integral + self.kd * derivative)
        angular = self.clamp(angular, -self.maxAngular, self.maxAngular)

        linear = self.getLinearSpeed(error)

        return linear, angular
    
    def clamp(self, value, minVal, maxVal):
        return max(min(value, maxVal), minVal)

    def getLinearSpeed(self, error):
        if abs(error) > 10:
            return self.linearVel * 0.3

        if abs(error) < 5:
            return self.linearVel * 1.5

        return self.linearVel
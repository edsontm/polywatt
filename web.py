import cgi
from watt import RiverPair
form = b'''

<html>
    <head>
        <title>Water Travel Time Estimator: Watt-E  </title>
    </head>
    <body>
        <center>
        <form method="post">
        <label>Polywatt testing area</label>
        <table border="0" cellspacing="2" cellpadding="2">
                <tbody><tr>
                    </tr>
                        <tr>
                        <td align="right">Stations:</td>
                        <td><input type="Text" name="TitleA" value="Upstream" size="14" maxlength="30"></td>
                        <td><input type="Text" name="TitleB" value="Downstream" size="14" maxlength="30"></td>
                        </tr>
                    <tr>
                        <td valign="top">Values:</td>
                            <td>
                            <textarea cols="12" rows="20" name="ColumnA" wrap="off"></textarea>
                            </td>
                            <td>
                            <textarea cols="12" rows="20" name="ColumnB" wrap="off"></textarea>
                            </td>
                    </tr>
                    <tr>
                        <td></td>
                        <td colspan="2" align="center">Enter one value per row.</td>
                    </tr>
                    <tr>
                    </tr>
                    <tr>
                    </tr>
                </tbody>
        </table>
                        <input type="submit" value="Run polywatt">
        </form>
        </center>
    </body>
</html>
'''

class EvalPairs():
    def __init__(self,upstream,downstream):
        self.up   = upstream
        self.down = downstream
        self.error_msg = ''

    def eval(self):
        ret = True
        vup   = self.upstream.split('\n')
        vdown = self.downstream.split('\n')
        if len(vup.split(',')) >1: # id_data mode
            mode = 'id_data' 
        else: # Data mode
            mode = 'data'
            self.rp = RiverPairs(pair_name="upstream_downstream")
            vup = np.array(vup)
            vup = vup.astype(np.float)
            vdown = np.array(vdown)
            vdown = vdown.astype(np.float)
            self.rp.set_raw_data(vup,vdown)
            first = 0
            print vup.shape
            


        return ret
    def run(self):
        pass


def test():
    a = '1,2,1,1,1,1,1,1'
    b = '1,1,1,1,1,2,1,1'


def app(environ, start_response):
    html = form

    if environ['REQUEST_METHOD'] == 'POST':
        post_env = environ.copy()
        post_env['QUERY_STRING'] = ''
        post = cgi.FieldStorage(
            fp=environ['wsgi.input'],
            environ=post_env,
            keep_blank_values=True
        )
        print post,post['TitleA'].value,"(",post['ColumnA'].value,","
        ev = EvalPairs(post['ColumnA'].value,post['ColumnB'].value)
        if (ev.eval()):
            ev.run()
        html += ev.error_msg


        
        #html = b'Hello, ' + post['name'].value + '!'

    start_response('200 OK', [('Content-Type', 'text/html')])
    return [html]

if __name__ == '__main__':
    if sys.argv[1] == 't':
        test()
    else:
        try:
            from wsgiref.simple_server import make_server
            httpd = make_server('', 8080, app)
            print('Serving on port 8080...')
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('Goodbye.')

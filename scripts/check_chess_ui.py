"""Browser checks for move input, reset, castling, promotion and checkmate."""
import json
import threading
from pathlib import Path
from playwright.sync_api import sync_playwright
from src.chess_web import make_server

class Policy:
    def move(self,moves):
        import chess
        board=chess.Board()
        for move in moves:board.push_uci(move)
        return next(iter(board.legal_moves)).uci()

def main():
    server=make_server('127.0.0.1',0,{'original':('Original large model',Path('runs/large/checkpoints/pretrain.pth'))},policies={'original':Policy()})
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url='http://127.0.0.1:'+str(server.server_port)
    screenshots=Path('runs/serving');screenshots.mkdir(parents=True,exist_ok=True)
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True,executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
            page=browser.new_page(viewport={'width':1100,'height':850})
            errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            def open_game(replies):
                page.unroute('**/api/move');queue=list(replies)
                def respond(route):
                    route.fulfill(status=200,content_type='application/json',body=json.dumps({'move':queue.pop(0)}))
                page.route('**/api/move',respond)
                page.goto(url);page.wait_for_function("document.querySelector('#status').textContent.includes('Your turn')")
            def move(start,end):
                page.locator('#board button').filter(has=page.locator('span')).count()
                page.locator('[aria-label^="'+start+' "]').click();page.locator('[aria-label^="'+end+' "]').click()
                page.wait_for_function("!document.querySelector('#status').textContent.includes('thinking')")
            open_game(['e7e5','b8c6','g8f6'])
            move('e2','e4');move('d1','h5');move('f1','c4');move('h5','f7')
            assert 'Checkmate. White wins.' in page.locator('#status').inner_text()
            page.locator('#new').click();assert page.locator('#moves li').count()==0
            open_game(['e7e5','b8c6','g8f6','f8c5'])
            move('e2','e4');move('g1','f3');move('f1','c4');move('e1','g1')
            assert page.locator('[aria-label="g1 white king"]').count()==1
            assert page.locator('[aria-label="f1 white rook"]').count()==1
            page.locator('#undo').click();assert page.locator('[aria-label="e1 white king"]').count()==1
            open_game(['h7h5','h5h4','h4h3','h3g2','g2h1q'])
            move('a2','a4');move('a4','a5');move('a5','a6');move('a6','b7')
            page.locator('[aria-label="b7 white pawn"]').click();page.locator('[aria-label="a8 black rook"]').click()
            page.locator('#promotion').wait_for(state='visible');page.get_by_role('button',name='Knight',exact=True).click()
            page.wait_for_function("!document.querySelector('#status').textContent.includes('thinking')")
            assert page.locator('[aria-label="a8 white knight"]').count()==1
            open_game(['e2e4','e2e4'])
            page.select_option('#side','b');page.wait_for_function("document.querySelector('#status').textContent.includes('Your turn')")
            assert page.locator('#undo').is_disabled()
            page.locator('#new').click()
            page.wait_for_function("document.querySelector('#status').textContent.includes('Your turn')")
            page.screenshot(path=str(screenshots/'chess-desktop.png'),full_page=True)
            page.set_viewport_size({'width':390,'height':850});page.select_option('#side','w')
            page.screenshot(path=str(screenshots/'chess-mobile.png'),full_page=True)
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
            assert not errors,errors
            browser.close()
            print('Browser checks passed: move input, checkmate, castling, undo, promotion, black-side opening, and responsive layout.')
    finally:
        server.shutdown();server.server_close();thread.join()

if __name__=='__main__':main()
